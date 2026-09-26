# Person 1 — CDM/DINA Design Document

## 1. Objective

The first stage of AdaptIQ creates an initial learner state. A student answers a diagnostic set of AI Engineering questions. We use a Cognitive Diagnostic Model (CDM), specifically a compact DINA-style model, to infer which latent skills the student likely possesses.

This is different from simply calculating a test score: the output is skill-specific mastery.

## 2. Skill ontology

| ID | Skill |
|---|---|
| S1 | Python & Data Handling |
| S2 | Machine Learning Fundamentals |
| S3 | Deep Learning |
| S4 | Data Preprocessing & Feature Engineering |
| S5 | Model Evaluation & Deployment |
| S6 | LLMs & Generative AI |

There are 30 diagnostic questions, approximately five questions per skill.

## 3. Q-matrix

The Q-matrix is a binary matrix with one row per question and one column per skill.

Q[i,j] = 1 means question i requires skill j.

Q[i,j] = 0 means question i does not require skill j.

Example:

```text
       S1 S2 S3 S4 S5 S6
Q1     1  0  0  0  0  0
Q28    0  0  1  0  0  1
Q30    1  0  0  0  0  1
```

Q28 deliberately requires both Deep Learning and LLM knowledge because self-attention is a Transformer concept used in modern LLMs. Q30 combines Python/data handling with GenAI because the question describes a RAG pipeline using embeddings and an LLM.

## 4. Why DINA?

A raw percentage does not tell the adaptive engine which underlying skills need work. DINA maps responses to latent mastery profiles using the Q-matrix.

There are K = 6 skills, so there are:

2^K = 64

possible binary mastery profiles.

The prototype evaluates all 64 profiles. This is computationally tiny for six skills and is easy to inspect and explain in a project demo.

## 5. DINA model

Let alpha_j be the learner's mastery of skill j:

alpha_j in {0,1}

For question i, let Q_ij indicate whether skill j is required.

The ideal response indicator is:

eta_i = product over j of alpha_j^(Q_ij)

Therefore eta_i = 1 only if the learner has every skill required by question i.

DINA then models:

P(X_i = 1 | eta_i = 1) = 1 - s_i

P(X_i = 1 | eta_i = 0) = g_i

where:
- s_i is the slip probability: knowing the required skills but answering incorrectly.
- g_i is the guess probability: lacking one or more required skills but answering correctly.

For this baseline prototype we use fixed:
- slip = 0.05
- guess = 0.20

These values can later be estimated/calibrated from a larger response dataset.

## 6. Inference

For each of the 64 mastery profiles alpha, calculate:

P(responses | alpha)

Using a uniform prior over profiles:

P(alpha | responses) proportional to P(responses | alpha)

The posterior is normalized across all profiles.

Two useful outputs are retained:

### MAP profile

The maximum a posteriori profile:

argmax_alpha P(alpha | responses)

This gives a binary interpretation such as:

S1=1, S2=1, S3=0, ...

### Marginal mastery probabilities

For each skill j:

P(alpha_j = 1 | responses)

This is more useful for the adaptive system because it preserves uncertainty.

Example:

```text
S1  0.94
S2  0.81
S3  0.27
S4  0.76
S5  0.34
S6  0.18
```

## 7. Why pass probabilities to Person 2?

Person 2 implements Knowledge Tracing. Passing probabilities instead of only 0/1 prevents information loss.

For example, these two students both have binary status S3=0, but:

Student A: P(S3 mastered)=0.49
Student B: P(S3 mastered)=0.02

They should not necessarily start BKT from the same state.

The initial mastery probabilities therefore form the interface between Person 1 and Person 2.

## 8. Limitations of this first prototype

1. Slip and guess are fixed rather than learned from calibration data.
2. The question bank is small.
3. The Q-matrix is manually authored.
4. Only six skills are modeled.
5. DINA assumes a conjunctive requirement: a multi-skill item requires all mapped skills.
6. The prototype uses binary latent mastery profiles.

These are acceptable for the mid-semester prototype. They also give clear directions for future improvement.

## 9. Person 2 interface

Person 2 should consume:

```python
mastery_probabilities = [
    P(S1 mastered),
    P(S2 mastered),
    P(S3 mastered),
    P(S4 mastered),
    P(S5 mastered),
    P(S6 mastered),
]
```

The order is always:

S1, S2, S3, S4, S5, S6.

Person 2 should not need to modify the DINA implementation.

## 10. End-to-end ownership

Person 1:
Question bank -> Q-matrix -> DINA -> initial mastery

Person 2:
Initial mastery -> BKT -> updated mastery after each response

Person 3:
Updated mastery -> adaptive question selection -> next question

This creates the project's central feedback loop.
