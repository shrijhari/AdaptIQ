# AdaptIQ — AI Engineering Adaptive Learning System

AdaptIQ is an adaptive learning system designed to estimate a learner's knowledge state and use that state to select appropriate learning questions.

## Goal

The system moves beyond a simple test score by estimating mastery of individual AI Engineering skills and adapting subsequent questions based on the learner's current knowledge state.

## End-to-End Pipeline

Student responses
        ↓
DINA / Cognitive Diagnostic Model
        ↓
Initial mastery probabilities
        ↓
Bayesian Knowledge Tracing (BKT)
        ↓
Updated mastery probabilities
        ↓
Adaptive Question Selection
        ↓
Next question
        ↓
Student response
        ↓
BKT update

## Project Components

### Person 1 — Diagnostic Assessment and Initial Learner Model

Person 1 developed the diagnostic assessment, Q-matrix and DINA-style Cognitive Diagnostic Model.

The DINA model estimates the learner's initial mastery probabilities across six AI Engineering skills.

Main files:

- `src/cdm.py`
- `data/questions.csv`
- `data/q_matrix.csv`
- `data/skills.csv`
- `data/student_responses.csv`
- `tests/test_cdm.py`

### Person 2 — Knowledge Tracing

Person 2 implemented Bayesian Knowledge Tracing (BKT) to update the learner's mastery after each response.

Main files:

- `src/bkt.py`
- `src/run_bkt.py`
- `src/plot_bkt.py`
- `data/bkt_history.csv`
- `data/bkt_mastery_trajectory.png`

### Person 3 — Adaptive Question Selection and Interface

Person 3 uses the updated mastery state from BKT to select the next question.

The adaptive component:

1. Identifies the weakest or most uncertain skill.
2. Creates a candidate pool of questions related to that skill.
3. Considers the question's skill relevance.
4. Considers the learner's current mastery and predicted correctness.
5. Considers question difficulty.
6. Selects the next question.
7. Sends the student's response back to BKT.
8. Uses the updated mastery state for the next selection.

Main files:

- `src/adaptive_engine.py`
- `src/adaptive_session.py`
- `src/adaptive_demo.py`
- `src/app.py`
- `tests/test_adaptive_engine.py`

## AI Engineering Skills

The system currently models six skills:

- S1: Python & Data Handling
- S2: Machine Learning Fundamentals
- S3: Deep Learning
- S4: Data Preprocessing & Feature Engineering
- S5: Model Evaluation & Deployment
- S6: LLMs & Generative AI

## Question Bank

The current diagnostic question bank contains 30 multiple-choice questions.

Each question has:

- Question text
- Answer options
- Correct answer
- Difficulty
- Skill mapping through the Q-matrix

The Q-matrix connects each question to the skills required to answer it.

## Adaptive Question Selection

The adaptive selector receives the learner's current mastery probabilities from BKT.

The selection process is:

Current mastery
        ↓
Identify weakest skill
        ↓
Find relevant questions
        ↓
Score candidate questions
        ↓
Select next question

The prototype scoring considers:

- Skill relevance
- Skill weakness
- Predicted correctness
- Difficulty match

## Interactive Prototype

A Streamlit interface is provided to demonstrate the adaptive learning loop.

The interface displays the learner's current mastery state, presents the selected question and updates the learner state after a response.

## Running the Project

Install the required dependencies:

```bash
pip install -r requirements.txt
