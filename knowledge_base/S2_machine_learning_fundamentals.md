# S2 --- Machine Learning Fundamentals

**Knowledge-base source:** *An Introduction to Statistical Learning*
(ISLP), 2023\
**Source file:** `ISLP_website.pdf`\
**Skill ID:** `S2`\
**Relevant source:** Chapter 2, Sections 2.1.4--2.2.3\
**Purpose:** Initial source material for AdaptIQ's RAG-based question
generation prototype.

> Page references identify both the PDF page and the printed book page
> where available. Preserve these references when chunking so generated
> questions can be traced to their source.

------------------------------------------------------------------------

## S2_C1 --- Supervised vs. Unsupervised Learning

**Source:** Chapter 2, Section 2.1.4\
**PDF pages:** 35--37\
**Printed book pages:** 25--27

In supervised learning, each observation includes predictor variables,
usually denoted by x_i, and a response variable, denoted by y_i. The aim
is to learn the relationship between the predictors and the response so
that we can make predictions or understand how the response depends on
the predictors.

In unsupervised learning, observations contain predictor variables but
no response variable. The goal is to learn about the structure or
relationships in the data. Clustering is an example of an unsupervised
learning method.

**Key distinction:** Supervised learning uses a known response variable
during learning; unsupervised learning does not.

------------------------------------------------------------------------

## S2_C2 --- Regression vs. Classification

**Source:** Chapter 2, Section 2.1.5\
**PDF page:** 37\
**Printed book page:** 27

The type of response variable determines whether a problem is a
regression or classification problem.

-   **Regression:** The response variable is quantitative.
-   **Classification:** The response variable is qualitative or
    categorical.

------------------------------------------------------------------------

## S2_C3 --- Measuring Quality of Fit: Mean Squared Error

**Source:** Chapter 2, Section 2.2.1\
**PDF pages:** 38--41\
**Printed book pages:** 28--31

For a regression problem, mean squared error (MSE) measures the average
squared difference between observed responses and predictions.

MSE = (1/n) × sum over i of (y_i − f_hat(x_i))²

Here, y_i is the observed response and f_hat(x_i) is the model's
prediction for observation i.

-   **Training MSE** is calculated using the observations used to fit
    the model.
-   **Test MSE** is calculated using observations that were not used to
    fit the model. It helps assess performance on unseen data.

A model with the smallest training MSE does not necessarily have the
smallest test MSE.

**Source note:** This section explains MSE for regression. It should not
be treated as a source for every possible machine-learning loss
function.

------------------------------------------------------------------------

## S2_C4 --- Training Error vs. Test Error in Classification

**Source:** Chapter 2, Section 2.2.3\
**PDF page:** 45\
**Printed book page:** 35

For classification:

-   **Training error rate** is the fraction of training observations
    that are incorrectly classified.
-   **Test error rate** is the fraction of test observations that are
    incorrectly classified.

A good classifier aims to have a low test error rate, because test
observations represent data not used to fit the model.

------------------------------------------------------------------------

## S2_C5 --- Overfitting

**Source:** Chapter 2, Section 2.2.1\
**PDF pages:** 38--41\
**Printed book pages:** 28--31

As model flexibility increases, training MSE generally decreases.
However, test MSE may decrease at first and then increase.

Overfitting occurs when a model fits the training data very closely,
including random noise, but performs poorly on unseen data. In this
situation, training MSE can be small while test MSE is large. A less
flexible model may have a smaller test MSE.

The discussion and illustration in Figure 2.9 show that minimizing
training MSE alone is not sufficient for choosing a model.

------------------------------------------------------------------------

## S2_C6 --- Bias--Variance Trade-Off

**Source:** Chapter 2, Section 2.2.2\
**PDF pages:** 42--44\
**Printed book pages:** 32--34

Expected test MSE can be decomposed into three components:

-   Variance
-   Squared bias
-   Irreducible error

More flexible models generally have lower bias but higher variance. The
aim is to balance bias and variance to obtain good test performance.

------------------------------------------------------------------------

## Source coverage and limitations

This initial S2 source extraction supports questions about:

-   Supervised and unsupervised learning
-   Regression and classification
-   Regression MSE
-   Training and test error
-   Overfitting
-   Bias--variance trade-off

It does **not** provide comprehensive coverage of all loss functions,
all machine-learning algorithms, or all model-evaluation metrics. Do not
generate questions on unsupported topics from this file alone.

## Chunk metadata

Use the following metadata when storing each section in the RAG index:

  Field             Value
  ----------------- -----------------------------------------------
  `skill_id`        `S2`
  `skill_name`      `Machine Learning Fundamentals`
  `source`          `ISLP_website.pdf`
  `chapter`         `2`
  `section_id`      Corresponding chunk ID, e.g. `S2_C5`
  `pdf_pages`       PDF page range listed for that chunk
  `printed_pages`   Printed book page range listed for that chunk
