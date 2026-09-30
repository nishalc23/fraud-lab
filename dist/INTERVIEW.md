# Discussing Fraud Lab

## A short explanation to practice

“I explored how to prioritize suspicious transactions under a limited review budget.
The current version uses simulated data. I compared logistic regression with gradient
boosting, kept training, calibration, validation and test in chronological order, and
selected the model using validation recall at a 1% review target. Logistic regression
won the validation tie-breaker and flagged 22 of 144 simulated test fraud cases with
23.4% precision. I built a demo that runs the exported models and shows how review
capacity, review cost and false alerts change the decision.”

Use this only after you have worked through the implementation. This version was
created with AI assistance; do not claim you independently wrote or deployed every
component, or that the result measures real financial impact.

## Why not accuracy?

Only 1.2% of the simulated test cases are fraud. Always predicting legitimate achieves
98.8% accuracy while catching no fraud. Average precision evaluates ranking; recall
and precision describe the actual review decision.

## Why a time split?

A future deployment scores later transactions. Random splitting mixes periods and
can obscure drift. The generator changes risk and new-device frequency with time.
The cutoff flagged 1% on validation and 0.783% on test, showing why a validation
capacity target is not a guaranteed future review volume.

## Why logistic regression?

It is a fast and understandable baseline. Gradient boosting allows nonlinear
interactions, but added complexity does not guarantee improvement. Both models had
12% validation recall at the original target; logistic had better validation average
precision. I did not choose the model from its test result.

## What does calibration do?

Balanced training deliberately changes the contribution of the rare class. Raw scores
can therefore be poor probability estimates. A separate unweighted logistic calibrator
maps raw logits to probabilities using a later partition at its natural prevalence.
Calibration can still fail under drift and should be checked on additional periods.

## What do the scores and explanations mean?

The browser uses exported coefficients, scaling parameters, decision trees and
calibration parameters. It produces the same scores as sklearn within numerical
precision on the checked vectors. The feature sensitivity is a one-at-a-time median
replacement, not a causal effect, SHAP value, or reason to automatically decline payment.

## Can you claim savings?

No realized savings are measured. The cost panel is a sensitivity analysis.
At the default hypothetical assumptions, prevention is 80% of flagged fraud amount,
review costs $3 per alert, and false alerts add $10 each. The formula is inspectable;
none of those assumptions is inferred from customer outcomes.

## What is weak about this project?

The label process and features are simulated. Events lack true account histories.
The experiment tests whether the pipeline can recover relationships created by a
generator, so it does not prove effectiveness against real adversaries. There are
only 144 test fraud cases, and recall is limited. Bootstrap intervals are conditional
and overlap. A real dataset, rolling windows and independent follow-up evaluation
would strengthen the work far more than decorating the demo.

## Before submitting

- Read `src/train.py` and explain each partition's purpose.
- Rerun the experiment and checks yourself.
- Change the review target and explain why test review rate differs.
- Identify what the simulator assumes and what it cannot establish.
- If the real dataset is added, update every result with the new run.

## Honest application draft for the current version

“Fraud Lab is an AI-assisted prototype I’m developing to explore transaction review
policies. The current version compares logistic regression and gradient boosting on
60,000 simulated transactions, with chronological fitting, calibration, validation
and test partitions. Model selection uses validation recall at a fixed review target,
and an interactive demo exposes the tradeoff between fraud recall, false alerts and
assumed review costs. The exported browser models are checked against Python outputs.
The next step is validation on real transaction data; current results are simulation-only.”
