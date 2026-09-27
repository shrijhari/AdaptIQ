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

## Question Banks

The system uses two separate question banks for different stages of the learning process.

### Diagnostic Question Bank

The diagnostic question bank contains 30 multiple-choice questions used during the initial learner assessment.

Each diagnostic question contains:

- Question text
- Answer options
- Correct answer
- Difficulty
- Skill mapping through the Q-matrix

The diagnostic questions are stored in:

- `data/questions.csv`
- `data/q_matrix.csv`

The diagnostic question bank is used by the DINA model to estimate the learner's initial mastery and by BKT to process the learner's diagnostic response history.

### Adaptive Question Bank

A separate adaptive question bank is used after the initial diagnostic and BKT stages.

The adaptive question bank contains 30 additional multiple-choice questions. These questions are kept separate from the diagnostic questions so that the adaptive learning stage does not simply repeat questions used during diagnosis.

The adaptive questions are stored in:

- `data/adaptive_questions.csv`
- `data/adaptive_q_matrix.csv`

The adaptive Q-matrix connects each adaptive question to the skills required to answer it.

The adaptive selector uses this separate question pool to select unseen questions based on the learner's current mastery state.

## Adaptive Question Selection

The adaptive selector receives the learner's current mastery probabilities from BKT.

The selection process is:

1. Identify the skill with the lowest current mastery.
2. Find unseen adaptive questions mapped to that skill.
3. Score candidate questions based on skill relevance and learner weakness.
4. Estimate predicted correctness from the learner's current mastery.
5. Match question difficulty to the learner's current mastery level.
6. Select the highest-scoring unseen adaptive question.
7. Present the question through the Streamlit interface.
8. Send the learner's response back to BKT.
9. Use the updated mastery state for the next question selection.

The selector prevents previously answered adaptive questions from being selected again unless repetition is explicitly enabled.

The adaptive selector is implemented in:

- `src/adaptive_engine.py`
- `src/adaptive_session.py`

The end-to-end demonstration is implemented in:

- `src/adaptive_demo.py`
- `src/app.py`

## Interactive Prototype

A Streamlit interface is provided to demonstrate the adaptive learning loop.

The interface displays the learner's current mastery state, presents the selected question and updates the learner state after a response.

## Running the Project

Install the required dependencies:

```bash
pip install -r requirements.txt

## **Run the Streamlit application:**

```bash
streamlit run src/app.py
