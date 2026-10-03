"""Online candidate producers sharing the secure artifact and evaluation lifecycle.

GRPO/RLOO use TRL's on-policy samplers. OPD/SDFT/SDPO sample the current
student inside every loss computation and score exactly those prefixes with a
frozen external or EMA self-teacher. No teacher-generated SFT dataset is written.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .artifacts import (
    ArtifactRef,
    EventLog,
    atomic_write_json,
    canonical_json_bytes,
    new_run_id,
    sha256_bytes,
    sha256_file,
    validate_run_card,
    verify_artifact_ref,
)
from .online import (
    online_method,
    online_rows,
    online_settings,
    self_teacher_prompt,
    split_online_rows,
    validate_online_policy,
    verify_response,
)
from .policy import Policy
from .resolver import CandidateRegistry, ResolvedModelRef


def reverse_kl(student_logits, teacher_logits, *, chunk_size: int):
    """Analytic per-token reverse KL on aligned student-generated prefixes."""
    import torch

    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("distillation logits must have aligned [tokens, vocabulary] shapes")
    if chunk_size <= 0 or student_logits.shape[0] == 0:
        raise ValueError("distillation requires non-empty tokens and a positive chunk size")
    losses, student_selected, teacher_selected = [], [], []
    for start in range(0, student_logits.shape[0], chunk_size):
        student = student_logits[start : start + chunk_size].float().log_softmax(-1)
        teacher = teacher_logits[start : start + chunk_size].detach().float().log_softmax(-1)
        losses.append((student.exp() * (student - teacher)).sum(-1))
        student_selected.append(student)
        teacher_selected.append(teacher)
    # The logs are used only by the caller to record probabilities of the actual
    # sampled tokens. Teacher targets are detached even if passed trainable tensors.
    return torch.cat(losses), student_selected, teacher_selected


def update_ema_teacher(student, teacher, decay: float) -> None:
    import torch

    student_parameters = dict(student.named_parameters())
    with torch.no_grad():
        for name, target in teacher.named_parameters():
            source = student_parameters[name]
            if source.requires_grad:
                target.mul_(decay).add_(source.detach(), alpha=1.0 - decay)


class RolloutJournal:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("x", encoding="utf-8")
        self.count = 0

    def record(self, value: Mapping[str, Any]) -> None:
        self.handle.write(canonical_json_bytes(value).decode() + "\n")
        self.handle.flush()
        self.count += 1

    def close(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()


def _encode(tokenizer, prompt, *, device, maximum: int):
    rendered = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    if encoded["input_ids"].shape[-1] > maximum:
        raise ValueError("online prompt exceeds its declared limit; truncation is forbidden")
    return {name: value.to(device) for name, value in encoded.items()}


def make_distillation_trainer(
    *,
    model,
    teacher,
    tokenizer,
    method: str,
    dataset,
    args,
    settings: Mapping[str, Any],
    journal: RolloutJournal,
    callbacks,
):
    if method not in {"sdpo", "sdft"}:
        raise ValueError("feedback-conditioned trainer supports only SDPO and SDFT")
    import torch
    from transformers import Trainer

    class OnlineDistillationTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if return_outputs:
                raise ValueError(
                    "online distillation evaluation uses the protected external evaluator"
                )
            device = next(model.parameters()).device
            losses = []
            for row in inputs["rows"]:
                encoded = _encode(
                    tokenizer, row["prompt"], device=device, maximum=settings["max_prompt_length"]
                )
                prefix_length = encoded["input_ids"].shape[-1]
                was_training = model.training
                model.eval()
                try:
                    with torch.no_grad():
                        generated = self.accelerator.unwrap_model(model).generate(
                            **encoded,
                            do_sample=True,
                            temperature=settings["temperature"],
                            top_p=1.0,
                            top_k=0,
                            min_new_tokens=settings["min_completion_length"],
                            max_new_tokens=settings["max_completion_length"],
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            use_cache=True,
                        )
                finally:
                    model.train(was_training)
                answer_ids = generated[:, prefix_length:]
                if not answer_ids.numel():
                    raise RuntimeError("online student generated no tokens")
                response = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
                teacher_prompt = self_teacher_prompt(row, method, response)
                teacher_encoded = _encode(
                    tokenizer,
                    teacher_prompt,
                    device=device,
                    maximum=self.args.max_teacher_context - answer_ids.shape[-1],
                )
                teacher_ids = torch.cat([teacher_encoded["input_ids"], answer_ids], dim=-1)
                # Only completion predictions enter the objective. Keeping prompt
                # logits would allocate context_length * vocabulary on both models.
                prediction_count = answer_ids.shape[-1] + 1
                with torch.no_grad():
                    teacher_logits = teacher(
                        input_ids=teacher_ids,
                        attention_mask=torch.ones_like(teacher_ids),
                        use_cache=False,
                        logits_to_keep=prediction_count,
                    ).logits[0, -prediction_count:-1]
                student_logits = model(
                    input_ids=generated,
                    attention_mask=torch.ones_like(generated),
                    use_cache=False,
                    logits_to_keep=prediction_count,
                ).logits[0, -prediction_count:-1]
                token_losses, student_logs, teacher_logs = reverse_kl(
                    student_logits,
                    teacher_logits,
                    chunk_size=settings["logit_chunk_size"],
                )
                if not torch.isfinite(token_losses).all():
                    raise RuntimeError("non-finite on-policy distillation loss")
                losses.append(token_losses.mean())
                sampled_student_logs, sampled_teacher_logs = [], []
                for index, (slog, tlog) in enumerate(zip(student_logs, teacher_logs)):
                    start = index * settings["logit_chunk_size"]
                    selected = answer_ids[0, start : start + slog.shape[0], None]
                    sampled_student_logs.extend(
                        slog.detach().gather(-1, selected).flatten().cpu().tolist()
                    )
                    sampled_teacher_logs.extend(tlog.gather(-1, selected).flatten().cpu().tolist())
                journal.record(
                    {
                        "method": method,
                        "rollout_policy_step": self.state.global_step,
                        "content_id": row["content_id"],
                        "prompt_ids": encoded["input_ids"][0].tolist(),
                        "completion_ids": answer_ids[0].tolist(),
                        "response": response,
                        "teacher_prompt": teacher_prompt,
                        "teacher_prompt_sha256": sha256_bytes(canonical_json_bytes(teacher_prompt)),
                        "student_log_probs": sampled_student_logs,
                        "teacher_log_probs": sampled_teacher_logs,
                        "token_reverse_kl": token_losses.detach().cpu().tolist(),
                    }
                )
            return torch.stack(losses).mean()

    trainer = OnlineDistillationTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=lambda rows: {"rows": rows},
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    # compute_loss averages each prompt; Trainer must apply accumulation scaling.
    trainer.model_accepts_loss_kwargs = False
    return trainer


def make_rl_trainer(
    *, model, tokenizer, method, train_dataset, eval_dataset, args, journal, callbacks
):
    from trl import GRPOTrainer, RLOOTrainer

    validation_ids = set(eval_dataset["content_id"])

    def verified_reward(
        completions,
        task_type,
        ground_truth,
        content_id,
        prompts,
        completion_ids,
        trainer_state,
        **_kwargs,
    ):
        lengths = [
            len(value)
            for value in (completions, task_type, ground_truth, content_id, prompts, completion_ids)
        ]
        if len(set(lengths)) != 1:
            raise ValueError("online reward inputs are not aligned")
        rewards = []
        for completion, kind, truth, identity, prompt, ids in zip(
            completions, task_type, ground_truth, content_id, prompts, completion_ids
        ):
            response = completion if isinstance(completion, str) else completion[-1]["content"]
            reward, feedback = verify_response(response, kind, truth)
            rewards.append(reward)
            journal.record(
                {
                    "method": method,
                    "phase": "validation" if identity in validation_ids else "train",
                    "rollout_policy_step": trainer_state.global_step,
                    "content_id": identity,
                    "prompt": prompt,
                    "completion_ids": ids,
                    "response": response,
                    "reward": reward,
                    "feedback": feedback,
                }
            )
        return rewards

    trainer_class = GRPOTrainer if method == "grpo" else RLOOTrainer
    return trainer_class(
        model=model,
        args=args,
        reward_funcs=verified_reward,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )


def make_opd_trainer(*, model, teacher, tokenizer, dataset, args, settings, journal, callbacks):
    """Use TRL's pinned on-policy sampler and GKD objective; add provenance only."""
    import torch
    from trl.experimental.gkd import GKDTrainer

    def collate(rows):
        ids = [
            _encode(tokenizer, row["prompt"], device="cpu", maximum=settings["max_prompt_length"])[
                "input_ids"
            ][0]
            for row in rows
        ]
        width = max(len(tokens) for tokens in ids)
        prompts = torch.full((len(ids), width), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(prompts)
        for index, tokens in enumerate(ids):
            prompts[index, -len(tokens) :] = tokens
            mask[index, -len(tokens) :] = 1
        return {
            "prompts": prompts,
            "prompt_attention_mask": mask,
            "input_ids": prompts.clone(),
            "attention_mask": mask.clone(),
            "labels": torch.full_like(prompts, -100),
            "content_ids": [row["content_id"] for row in rows],
        }

    class RecordedGKDTrainer(GKDTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if inputs["input_ids"].shape[-1] > teacher.config.max_position_embeddings:
                raise ValueError("OPD rollout exceeds teacher context; truncation is forbidden")
            captured = {}
            hook = self.teacher_model.register_forward_hook(
                lambda _model, _inputs, output: captured.update(logits=output.logits.detach())
            )
            try:
                # The Trainer counts num_items_in_batch from the collated labels, which are all
                # -100 before on-policy generation, so it is 0. TRL 1.x GKD divides the loss by
                # it (inf/NaN); TRL 0.23 ignored it and averaged over the generated completion
                # tokens. Passing None keeps that per-token mean; the Trainer's own gradient
                # accumulation scaling still receives the original value, as before.
                loss, output = super().compute_loss(
                    model, inputs, return_outputs=True, num_items_in_batch=None
                )
                if not torch.isfinite(loss).all():
                    raise RuntimeError("non-finite on-policy distillation loss")
                prefix = inputs["prompts"].shape[1]
                with torch.no_grad():
                    for index, identity in enumerate(inputs["content_ids"]):
                        valid = inputs["labels"][index, prefix:] != -100
                        completion = inputs["input_ids"][index, prefix:][valid]
                        if not completion.numel():
                            raise RuntimeError("online student generated no trainable tokens")
                        student_logits = output.logits[index, prefix - 1 : -1][valid].detach()
                        teacher_logits = captured["logits"][index, prefix - 1 : -1][valid]
                        kl, student_logs, teacher_logs = reverse_kl(
                            student_logits, teacher_logits, chunk_size=settings["logit_chunk_size"]
                        )
                        selected_student, selected_teacher = [], []
                        for chunk, (slog, tlog) in enumerate(zip(student_logs, teacher_logs)):
                            start = chunk * settings["logit_chunk_size"]
                            tokens = completion[start : start + slog.shape[0], None]
                            selected_student.extend(
                                slog.gather(-1, tokens).flatten().cpu().tolist()
                            )
                            selected_teacher.extend(
                                tlog.gather(-1, tokens).flatten().cpu().tolist()
                            )
                        journal.record(
                            {
                                "method": "opd",
                                "trainer": "trl.GKDTrainer",
                                "rollout_policy_step": self.state.global_step,
                                "content_id": identity,
                                "prompt_ids": inputs["prompts"][index][
                                    inputs["prompt_attention_mask"][index].bool()
                                ].tolist(),
                                "completion_ids": completion.tolist(),
                                "response": tokenizer.decode(completion, skip_special_tokens=True),
                                "student_log_probs": selected_student,
                                "teacher_log_probs": selected_teacher,
                                "token_reverse_kl": kl.cpu().tolist(),
                            }
                        )
                return (loss, output) if return_outputs else loss
            finally:
                hook.remove()
                captured.clear()

    trainer = RecordedGKDTrainer(
        model=model,
        teacher_model=teacher,
        args=args,
        data_collator=collate,
        processing_class=tokenizer,
        train_dataset=dataset,
        callbacks=callbacks,
    )
    trainer.generation_config.min_new_tokens = settings["min_completion_length"]
    return trainer


def _published_ref(source: Path, destination: Path, *, role: str, root: Path) -> dict[str, Any]:
    media_type = (
        "application/vnd.boldt.peft-adapter"
        if role == "adapter_checkpoint"
        else "application/octet-stream"
    )
    measured = ArtifactRef.from_path(source, role=role, media_type=media_type)
    try:
        path = destination.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        path = str(destination.resolve())
    return ArtifactRef(
        path, measured.kind, role, measured.sha256, measured.size_bytes, measured.media_type
    ).to_dict()


def train_online_candidate(
    *,
    config: Mapping[str, Any],
    policy: Policy,
    manifest: Mapping[str, Any],
    outputs_root: Path,
    repository_root: Path,
    deadline: float,
    allow_gpu: bool,
    allow_checkpoints: bool,
    teacher_ref: ResolvedModelRef | None = None,
    teacher_license: str | None = None,
) -> dict[str, Any]:
    if not (allow_gpu and allow_checkpoints):
        raise ValueError("online training requires --allow-gpu --allow-checkpoints")
    validate_online_policy(config, policy)
    if (
        manifest.get("status") != "trainable"
        or manifest.get("policy_sha256") != sha256_file(policy.path)
        or manifest.get("license_status") != "usable"
        or manifest.get("leakage_statistics") != {"status": "clean", "hit_count": 0}
    ):
        raise ValueError(
            "online training requires a policy-matching, licensed, leakage-clean manifest"
        )
    method = online_method(config["experiment"]["lever"])
    settings = online_settings(config)
    rows = online_rows(manifest, config, root=repository_root)
    if time.monotonic() >= deadline:
        raise RuntimeError("online budget exhausted before model loading")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
    from transformers import TrainerCallback, TrainingArguments, set_seed
    from trl import GRPOConfig, RLOOConfig
    from trl.experimental.gkd import GKDConfig

    from .secure_compat import provenance
    from .secure_compat.training import (
        collect_model_metadata,
        create_model_and_tokenizer,
        validate_target_modules,
        validate_tokenizer,
    )
    from .training import warmup_steps_from_ratio

    if not torch.cuda.is_available():
        raise RuntimeError("online candidate training requires CUDA; CPU fallback is forbidden")
    training = config["training"]
    device = "cuda:0"
    torch.cuda.set_device(0)
    set_seed(training["seed"])
    source, revision = policy.seed_model["repo_id"], policy.seed_model["revision"]
    qlora = training["method"] == "qlora"
    run_id = new_run_id(f"train-{method}")
    checkpoint_staging = outputs_root / "checkpoints/.staging" / run_id
    checkpoint_final = outputs_root / "checkpoints" / run_id
    run_staging = outputs_root / "runs/.staging" / run_id
    run_final = outputs_root / "runs" / run_id
    events = EventLog(outputs_root)
    run_type = "train_rlvr" if method == "rloo" else f"train_{method}"
    started = time.monotonic()
    start = events.append("run_started", run_id, {"run_type": run_type})
    journal = RolloutJournal(run_staging / "rollouts.jsonl")
    callback_exhausted = False
    try:
        model, tokenizer = create_model_and_tokenizer(
            source,
            revision=revision,
            qlora=qlora,
            gradient_checkpointing=training["gradient_checkpointing"],
            policy=policy,
            device=device,
        )
        validate_tokenizer(tokenizer, training["context_length"], model)
        metadata = collect_model_metadata(source, revision, model, policy)
        validate_target_modules(model, training["target_modules"])
        lora = LoraConfig(
            task_type="CAUSAL_LM",
            r=training["lora_r"],
            lora_alpha=training["lora_alpha"],
            lora_dropout=training["lora_dropout"],
            target_modules=training["target_modules"],
            bias="none",
        )
        model = get_peft_model(model, lora)
        if training["gradient_checkpointing"]:
            model.enable_input_require_grads()
        for row in rows:
            _encode(tokenizer, row["prompt"], device=device, maximum=settings["max_prompt_length"])
        teacher = None
        if method == "opd":
            if (
                teacher_ref is None
                or teacher_license not in policy.document["data"]["allowed_licenses"]
            ):
                raise ValueError("OPD requires an exact, resolved teacher with a usable license")
            if (
                teacher_ref.tokenizer_sha256 != metadata["tokenizer_sha256"]
                or teacher_ref.chat_template_sha256 != metadata["chat_template_sha256"]
            ):
                raise ValueError(
                    "OPD requires the exact student tokenizer and chat template; cross-tokenizer fallback is forbidden"
                )
            from .secure_compat.evaluation import load_transformers_model

            if teacher_ref.artifact:
                verify_artifact_ref(teacher_ref.artifact, root=repository_root)
            teacher, teacher_tokenizer = load_transformers_model(teacher_ref, device=device)
            if teacher_tokenizer.get_vocab() != tokenizer.get_vocab():
                raise ValueError("OPD teacher/student vocabularies are not aligned")
        elif method in {"sdpo", "sdft"}:
            teacher_base, _ = create_model_and_tokenizer(
                source,
                revision=revision,
                qlora=qlora,
                gradient_checkpointing=False,
                policy=policy,
                device=device,
            )
            teacher = get_peft_model(teacher_base, lora)
            teacher.load_state_dict(model.state_dict())
            del teacher_base
        if teacher is not None:
            teacher.eval()
            teacher.requires_grad_(False)
        # Finish/save/reload reserve is inside the one global loop deadline.
        remaining = deadline - time.monotonic()
        reserve = min(300.0, max(15.0, remaining * 0.1))
        if remaining <= reserve:
            raise RuntimeError("online budget cannot accommodate checkpoint save/reload")

        class OnlineStepCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                nonlocal callback_exhausted
                if method in {"sdpo", "sdft"}:
                    update_ema_teacher(kwargs["model"], teacher, settings["teacher_ema_decay"])
                if time.monotonic() >= deadline - reserve:
                    callback_exhausted = True
                    control.should_training_stop = True
                return control

        common = {
            "output_dir": str(run_staging / "trainer"),
            "max_steps": training["max_steps"],
            "num_train_epochs": training["num_train_epochs"],
            "learning_rate": training["learning_rate"],
            # Transformers 5 removed warmup_ratio; warmup_steps in [0, 1) is the same ratio.
            "warmup_steps": warmup_steps_from_ratio(
                training["warmup_ratio"], training["max_steps"]
            ),
            "per_device_train_batch_size": settings["batch_size"],
            "gradient_accumulation_steps": training["gradient_accumulation_steps"],
            "gradient_checkpointing": training["gradient_checkpointing"],
            "max_grad_norm": settings["max_grad_norm"],
            "seed": training["seed"],
            "data_seed": training["seed"],
            "bf16": True,
            "use_cpu": False,
            "save_strategy": "no",
            "report_to": "none",
            "logging_steps": 1,
            "remove_unused_columns": False,
            "logging_nan_inf_filter": False,
        }
        splits = None
        if method in {"grpo", "rloo"}:
            train_rows, validation_rows = split_online_rows(
                rows, settings["validation_fraction"], training["seed"]
            )
            splits = {
                "train": [r["content_id"] for r in train_rows],
                "validation": [r["content_id"] for r in validation_rows],
            }
            options = {
                **common,
                "per_device_eval_batch_size": settings["batch_size"],
                "num_generations": settings["num_generations"],
                # TRL 1.x removed max_prompt_length and never truncates prompts; every
                # prompt was already checked against it by _encode, which fails closed.
                "max_completion_length": settings["max_completion_length"],
                "temperature": settings["temperature"],
                "beta": settings["beta"],
                "use_vllm": False,
                "use_liger_kernel": False,
                "num_iterations": 1,
                "steps_per_generation": 1,
                "mask_truncated_completions": True,
            }
            if method == "grpo":
                options.update(
                    loss_type=settings["loss_type"], scale_rewards=settings["scale_rewards"]
                )
            args = (GRPOConfig if method == "grpo" else RLOOConfig)(**options)
            args._n_gpu = 1
            trainer = make_rl_trainer(
                model=model,
                tokenizer=tokenizer,
                method=method,
                train_dataset=Dataset.from_list(train_rows),
                eval_dataset=Dataset.from_list(validation_rows),
                args=args,
                journal=journal,
                callbacks=[OnlineStepCallback()],
            )
        elif method == "opd":
            args = GKDConfig(
                **common,
                lmbda=1.0,
                beta=1.0,
                seq_kd=False,
                disable_dropout=True,
                max_length=training["context_length"],
                max_new_tokens=settings["max_completion_length"],
                temperature=settings["temperature"],
                packing=False,
            )
            args._n_gpu = 1
            trainer = make_opd_trainer(
                model=model,
                teacher=teacher,
                tokenizer=tokenizer,
                dataset=Dataset.from_list(rows),
                args=args,
                settings=settings,
                journal=journal,
                callbacks=[OnlineStepCallback()],
            )
        else:
            args = TrainingArguments(**common)
            args._n_gpu = 1
            args.max_teacher_context = min(
                training["context_length"], teacher.config.max_position_embeddings
            )
            trainer = make_distillation_trainer(
                model=model,
                teacher=teacher,
                tokenizer=tokenizer,
                method=method,
                dataset=Dataset.from_list(rows),
                args=args,
                settings=settings,
                journal=journal,
                callbacks=[OnlineStepCallback()],
            )
        trainer.model.is_parallelizable = True
        trainer.model.model_parallel = True
        result = trainer.train()
        if trainer.state.global_step < 1:
            raise RuntimeError("online training completed no optimizer step")
        if not torch.isfinite(torch.tensor(result.training_loss)):
            raise RuntimeError("online training produced non-finite loss")
        validation_metrics = trainer.evaluate() if method in {"grpo", "rloo"} else {}
        checkpoint_staging.mkdir(parents=True)
        trainer.save_model(str(checkpoint_staging))
        tokenizer.save_pretrained(checkpoint_staging)
        if teacher is not None and method in {"sdpo", "sdft"}:
            from safetensors.torch import save_file

            save_file(
                {
                    k: v.detach().cpu().contiguous()
                    for k, v in get_peft_model_state_dict(teacher).items()
                },
                str(run_staging / "teacher_ema.safetensors"),
            )
        steps = int(trainer.state.global_step)
        history = trainer.state.log_history
        # Reload on the same explicit target device, rather than using CPU smoke.
        del trainer, model
        teacher = None
        torch.cuda.empty_cache()
        reload_base, reload_tokenizer = create_model_and_tokenizer(
            source,
            revision=revision,
            qlora=qlora,
            gradient_checkpointing=False,
            policy=policy,
            device=device,
        )
        reloaded = PeftModel.from_pretrained(reload_base, str(checkpoint_staging))
        probe = _encode(
            reload_tokenizer,
            rows[0]["prompt"],
            device=device,
            maximum=settings["max_prompt_length"],
        )
        with torch.no_grad():
            logits = reloaded(**probe, logits_to_keep=1).logits
        if not torch.isfinite(logits).all():
            raise RuntimeError("online checkpoint reload produced non-finite logits")
        del reloaded, reload_base
        if time.monotonic() >= deadline:
            raise RuntimeError("online budget exhausted during validation/save/reload")
        journal.close()
        duration = time.monotonic() - started
        metrics = {
            "train_loss": float(result.training_loss),
            "steps_completed": steps,
            "rollouts_recorded": journal.count,
            "wall_clock_seconds": duration,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
            "stop_reason": "budget_limit" if callback_exhausted else "max_steps",
            "validation": validation_metrics,
            "log_history": history,
        }
        parameters = {
            "method": method,
            "training": dict(training),
            "online": settings,
            "rollout_source": "current_student",
            "trainer": "trl.GKDTrainer"
            if method == "opd"
            else (
                "trl.GRPOTrainer"
                if method == "grpo"
                else "trl.RLOOTrainer"
                if method == "rloo"
                else "feedback_conditioned_trainer"
            ),
            "teacher": teacher_ref.to_dict() if teacher_ref else None,
            "teacher_license": teacher_license,
            "self_teacher": "ema" if method in {"sdpo", "sdft"} else None,
            "splits": splits,
        }
        outputs = [
            _published_ref(
                checkpoint_staging,
                checkpoint_final,
                role="adapter_checkpoint",
                root=repository_root,
            ),
            _published_ref(
                run_staging / "rollouts.jsonl",
                run_final / "rollouts.jsonl",
                role="online_rollouts",
                root=repository_root,
            ),
        ]
        if (run_staging / "teacher_ema.safetensors").exists():
            outputs.append(
                _published_ref(
                    run_staging / "teacher_ema.safetensors",
                    run_final / "teacher_ema.safetensors",
                    role="teacher_ema",
                    root=repository_root,
                )
            )
        inputs = [*manifest["shards"], *manifest.get("reports", [])]
        if teacher_ref is not None and teacher_ref.artifact:
            inputs.append(teacher_ref.artifact)
        card = {
            "schema_version": 1,
            "run_id": run_id,
            "run_type": run_type,
            "mode": "real",
            "status": "succeeded",
            "started_at": start["event"]["timestamp"],
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": duration,
            "command": [
                "uv",
                "run",
                "--locked",
                "python",
                "-m",
                "boldt_posttrain.cli",
                "train",
                config["experiment"]["lever"],
                "--real",
            ],
            "git": provenance.collect_git("HEAD", root=repository_root),
            "policy": {"path": str(policy.path), "sha256": sha256_file(policy.path)},
            "experiment": {
                "path": "in-memory",
                "sha256": sha256_bytes(canonical_json_bytes(config)),
                "resolved_sha256": sha256_bytes(canonical_json_bytes(parameters)),
            },
            "inputs": inputs,
            "outputs": outputs,
            "model": metadata,
            "data": dict(manifest),
            "parameters": parameters,
            "hardware": provenance.collect_hardware(),
            "environment": {
                **provenance.collect_environment(),
                "metrics": metrics,
                "event_head": {
                    key: start[key] for key in ("sequence", "last_event_hash", "log_sha256")
                },
            },
            "parents": [teacher_ref.source_run_id]
            if teacher_ref is not None and teacher_ref.source_run_id
            else [],
            "compatibility_fingerprint": sha256_bytes(canonical_json_bytes(metadata)),
            "error": None,
        }
        validate_run_card(card)
        atomic_write_json(run_staging / "run_card.json", card)
        os.replace(checkpoint_staging, checkpoint_final)
        os.replace(run_staging, run_final)
        events.append(
            "run_finished",
            run_id,
            {"status": "succeeded", "run_card_sha256": sha256_file(run_final / "run_card.json")},
        )
        CandidateRegistry(outputs_root).rebuild(policy)
        return {
            "status": "succeeded",
            "run_id": run_id,
            "checkpoint": str(checkpoint_final),
            "method": method,
            "metrics": metrics,
        }
    except Exception as exc:
        if not journal.handle.closed:
            journal.close()
        events.append(
            "run_finished", run_id, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        )
        raise
