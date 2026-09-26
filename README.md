# AdaptIQ — Person 1: Initial Learner Model

## Goal
Estimate a learner's initial mastery of six AI Engineering skills from diagnostic question responses.

## Pipeline
Student responses -> Q-matrix -> DINA/CDM -> initial mastery probabilities -> Person 2 (BKT)

## Skills
- S1: Python & Data Handling
- S2: Machine Learning Fundamentals
- S3: Deep Learning
- S4: Data Preprocessing & Feature Engineering
- S5: Model Evaluation & Deployment
- S6: LLMs & Generative AI

## Run
From the project root:

```bash
python src/cdm.py
```

Run tests:

```bash
python -m pytest
```

If pytest is unavailable, the model itself can still be run with the first command.

## DINA parameters
Baseline demo values:
- slip = 0.05
- guess = 0.20

These are configurable and are not learned from a large calibration dataset in this first prototype.

## Handoff to Person 2
The important output is:

`mastery_probabilities`

This is a six-element vector containing P(skill mastered | observed responses). Person 2 can use these values as the initial learner state for BKT.
