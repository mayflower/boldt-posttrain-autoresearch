# Scoring and promotion

Scoring requires zero technical errors, verified leakage/license evidence, complete lm-eval tasks,
and all hard safety/language/format gates. German language retention has its own score weight, delta
gate, and paired bootstrap interval.

Promotion requires a `promotion` candidate against the immutable promotion baseline. Gates use the
lower confidence bounds for German instruction, safety, and language retention, and upper bounds
for over-refusal and English bleed. A dev score alone cannot promote.

The general frontier remains strict. Separate explicit reasoning, coding, format, long-context, and
safety frontiers may advance after common hard gates when their own dimension improves. Verified
specialist frontier entries are eligible merge inputs.
