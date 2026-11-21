# 15.071 course checklist: model-building and evaluation recommendations

Extracted from the 8 lecture decks in `docs/course/` (sessions 2-9). Organized by
lecture; each item cites the lecture and, where legible, the slide number. This is a
reference for reviewing `src/` and `notebooks/` against the course's stated best
practices -- it is not itself a judgment of our pipeline (see the separate review).

Decks are not committed to git (copyrighted course material); this checklist and its
citations are safe to share and commit on their own.

---

## Lecture 2 (S02, 2026-09-14) -- Linear regression (solar power forecasting)

**Train/test splits**
- Set aside a test set, typically 20-50% of the data (p12).
- Three split strategies, pick based on the data's structure (p12):
  - **Temporal split**: hold out the most recent period, to replicate "learning from
    the past, predicting the future" -- used when the task has a time horizon and
    seasonality.
  - **Random split**: a random test set.
  - **Stratified sampling**: random split constructed to match distributions between
    train and test.
- Keep the test set hands-off until final evaluation, "to avoid contamination and
  test the model's true predictive power" (p12).

**Baselines**
- R² is computed against a baseline that predicts the sample average on the training
  data, with no feature information (p24).

**Metrics**
- Out-of-sample R² (OSR²): out-of-sample improvement over the **train-set mean**
  baseline, evaluated on test actuals (p35).
- MAE and RMSE, in the outcome's natural units (p35).
- **R² should not replace data visualizations, managerial insight, and common sense**
  (p25) -- a high R² can still mislead if the fit is driven by a few points or a
  structural break.
- **Statistical significance of a coefficient does not mean the feature improves
  predictive performance** -- Model 1's significant coefficients did not necessarily
  translate into out-of-sample gains; Model 2's temporal variables were more
  predictive than the weather variables despite both being "significant" (p39).

**Cross-validation / regularization**
- Not covered in this lecture (introduced in L4).

**Interpretability**
- Coefficients are each a marginal effect ("every unit increase in X is expected to
  increase Y by (beta)"), with significance tests (p23).

**Business framing**
- A poll explicitly tests the train/test discipline: "the goal of machine learning
  is to maximize performance on the test set, not the training set" is a listed
  (correct) statement, and "a model that achieves strong training performance will
  also achieve strong testing performance" is a listed (incorrect) distractor (p37).
- Translate predictions into an operational decision under asymmetric costs (the
  newsvendor problem): bid lower than the model's point prediction when under-supply
  and over-supply costs differ, and quantify the resulting profit/margin change
  against the "current" (unmodeled) strategy (p37-40).

---

## Lecture 3 (S03, 2026-09-16) -- Logistic regression (customer churn)

**Train/test splits**
- Random 70/30 split, **stratified so the training and test sets have the same
  churn rate** (class balance preserved across the split) (p9).

**Baselines**
- For an imbalanced outcome, state the **naive majority-class baseline** explicitly
  (predicting "no churn" for everyone achieves 73% accuracy) before claiming any
  model is "better" -- and ask what "better" should even mean here (p10).

**Metrics**
- Confusion-matrix quartet (TN/TP/FP/FN), each given a business meaning specific to
  the use case (cost of an unnecessary intervention vs. a missed at-risk customer)
  (p23).
- Accuracy, True Positive Rate (sensitivity), False Positive Rate, Precision --
  formulas given explicitly (p24).
- ROC curve (TPR vs. FPR across all thresholds) and AUC: unit-free, threshold-free
  summary of discrimination; AUC = 0.5 for a random/baseline classifier, 1.0 for a
  perfect one (p26-27).
- "Different metrics accomplish different goals: accuracy is simple and
  interpretable but sensitive to the classification threshold and to class balance;
  AUC is threshold-independent" (p28).

**Cross-validation / regularization**
- Not covered in this lecture.

**Interpretability**
- Coefficient sign and the exponential-of-coefficient odds ratio, read directly as
  "more/less likely to churn" (p17).
- Ad hoc model simplification: drop an insignificant, collinear feature
  (`MonthlyCharges`, correlated with `InternetService`); re-bucket a categorical
  level that behaves differently from the rest (p19).

**Business framing**
- The **classification threshold should be chosen to maximize business profit, not
  defaulted to 0.5** -- worked example shows the profit-optimal threshold (0.33 here)
  differs from 0.5, and that picking the data-driven optimum over the default 0.5
  captures an extra ~1 point of profit gain (5.8% vs. 4.9%) over a naive uniform
  policy (which loses 5.5%) (p31-35).
- Expected-profit framing: assign a dollar payoff to each confusion-matrix cell
  (true negative, true positive, false positive, false negative) and choose the
  decision rule (here, the discount threshold) that maximizes expected profit, not
  the rule that maximizes a classification metric (p31-32).

---

## Lecture 4 (S04, 2026-09-21) -- Model selection: regularization & cross-validation (Moneyball salaries)

**Train/test splits**
- 70/30 split (p7).
- **A linear/unregularized model maximizes in-sample fit but is not guaranteed to
  maximize out-of-sample performance** -- this is the motivation for everything that
  follows in this lecture (p12).

**Baselines**
- Plain (unregularized) linear regression using all available variables is itself
  treated as a baseline to beat with regularized models (p26, p33: "linear
  regression" row kept in every results table for comparison).

**Metrics**
- **Always report in-sample AND out-of-sample R²/RMSE side by side**, for every
  candidate model, to diagnose over/underfitting visually (p26, p33, p35-36).
- Cross-validated error is "pessimistic" (each fold trains on less data than the
  full training set): after hyperparameter selection, **refit the chosen
  configuration on the full training set** before reporting final out-of-sample
  performance (p36).

**Cross-validation / regularization**
- **Multicollinearity** diagnosis via a correlation matrix: "a model does not know
  whether the outcome stems from one variable or another" when predictors are
  highly correlated -- this is the core motivation for regularization here (p9-10).
- **Underfitting vs. overfitting** as a U-shaped curve in model complexity vs. error
  on unseen data; "model selection: solving the overfitting vs. underfitting
  trade-off using the training data only but never using the test data" (p13-14).
- **Ridge (L2)** penalizes squared coefficient magnitude (shrinks, rarely to zero);
  **LASSO (L1)** penalizes absolute magnitude and performs variable selection
  (shrinks exactly to zero). Ridge handles correlated variables better; LASSO yields
  a sparser, more interpretable model (p15-16).
- **"These [shrinkage] methods have no convenient test of statistical
  significance"** -- regularized coefficients should not be read with p-values the
  way OLS coefficients are (p16).
- **Standardize features before regularizing** (subtract mean, divide by std),
  because penalty magnitude is scale-sensitive -- and **compute the mean/std from the
  training data only, to avoid data leakage from the test set**. Same procedure used
  for neural networks later in the course (p21).
- **A single validation split gives unstable (high-variance) hyperparameter choices**
  -- re-running the random split changes the selected lambda noticeably (p29).
  **K-fold cross-validation** (here, 10-fold) is the fix: fit on k-1 folds, score on
  the held-out fold, average the loss across folds, repeat per candidate
  hyperparameter, then **refit the winning hyperparameter on the entire training
  set** (p30-31, p36).
- Explicit code pattern: score on the validation fold, never the test set; refit on
  the full training set with the selected hyperparameter afterward (p31).
- **"Surviving LASSO coefficients do not indicate 'significant' variables"** --
  variable selection by regularization is not a hypothesis test (p34).

**Interpretability**
- Hyperparameter = "a regularization weight that controls the strength of the
  penalty -- the larger the penalty, the simpler the model" (p39).

**Business framing**
- Poll explicitly tests which justification for model choice is correct among:
  best training fit, highest out-of-sample R², highest cross-validated R², lowest
  cross-validated RMSE (p35) -- the lesson is that the right justification is the one
  tied to genuine out-of-sample / cross-validated generalization, not training fit.
- "Model selection results in close-to-optimal out-of-sample performance" is the
  stated impact of doing all of the above correctly (p39).

---

## Lecture 5 (S05, 2026-09-23) -- CART: classification and regression trees (healthcare claims costs)

**Train/test splits**
- 70/30 split (p7). Heavily right-skewed monetary outcome (medical reimbursement)
  is **log-transformed** (`log10(1+reimbursement)`) before modeling (p9).

**Baselines**
- Two explicit baselines reported alongside every model: a **"historical baseline"**
  (next year's cost = this year's cost) and an **unregularized linear regression
  using all variables** (noting its own multicollinearity issues) (p33).

**Metrics**
- R², out-of-sample R² (OSR²), RMSE (p35 table), reported on the original dollar
  scale even when the model was fit on a log scale -- "report errors in original,
  natural units, even when modeling choices lead to derived quantities" (p34).
- **Calibration**: "predicted total out of total actual" is an explicit column in
  the model-assessment table, alongside interpretability and complexity -- i.e.,
  *whether the sum of predictions matches the sum of actuals* is treated as its own
  assessment criterion, separate from per-observation error (p35).

**Cross-validation / regularization**
- CART's regularization-equivalent hyperparameters: `ccp_alpha` (cost-complexity
  pruning), `min_samples_leaf`, `max_depth` -- "main parameters to control
  underfitting vs. overfitting" (p24).
- **Jointly cross-validate the hyperparameters** (fix one, relax/search the others,
  or search all three together) via k-fold CV; pick the complexity that minimizes
  average cross-validated loss, which sits between visibly overfit and visibly
  underfit configurations (p28-30).
- **A CART model literally cannot extrapolate: "every prediction is an average of
  observations, so a tree can never predict a value outside the range in the
  training data"** -- listed as a named limitation of trees, alongside lack of
  smoothness (small feature changes can cause large prediction jumps) and
  instability (retraining on similar data can produce a very different tree) (p18).

**Interpretability**
- Threshold-based decision rules are easy to follow and explain to non-technical
  stakeholders ("managers, clinicians, regulators, customers") -- listed as CART's
  main advantage over linear/logistic models (p11).
- CART performs implicit **variable selection** (unused features simply never get
  split on) (p11).
- The variables nearest the root of the tree are "usually the most impactful"
  (p11).

**Business framing**
- Translate predicted cost into a managerial decision (flag the top-X% highest-
  predicted-cost patients for a case-management program), with an explicit
  cost/savings assumption set, and report **hit rate, net savings, and a break-even
  curve over the number of patients flagged** (p36-38). A simple ("small") tree can
  lose real value vs. a cross-validated one in this net-savings curve (p38).

---

## Lecture 6 (S06, 2026-09-28) -- Tree ensembles: random forests & boosting (wildfire growth)

**Train/test splits**
- **Temporal split** explicitly chosen "to avoid data leakage from future fire
  seasons": train on 2015-2022, test on 2023-2024 (p13).
- "**Cross-validation will group [related] entries to avoid leakage**" -- when
  rows share structure (e.g., multiple observations from the same fire event),
  CV folds must respect that grouping, not split it randomly (p13).

**Baselines / metrics**
- Final comparison reports **train R² and test R² side by side** across linear
  regression, a single tree, random forest, and boosting, to show both the size of
  the ensemble benefit and the smaller additional benefit from cross-validated
  (vs. default) hyperparameters (p40).

**Cross-validation / regularization**
- Random forest mechanics: bootstrap-resampled training sets + a random feature
  subset per split, averaged across trees ("wisdom of crowds") to reduce variance
  (p18-20).
- Random forest hyperparameters: number of trees, model complexity (max depth / min
  observations per leaf), tree diversity (max features considered per split) (p22).
  Out-of-sample performance has a "sweet spot" in complexity and diversity; more
  trees alone rarely overfits a random forest (p23).
- Boosting mechanics: many *shallow, weak* trees, fit **sequentially** on the
  residuals of the current ensemble, each new tree's contribution shrunk by a
  **learning rate** (slow learning) (p28-29).
- Boosting hyperparameters: number of trees, complexity (depth, min samples),
  **learning rate** -- with "strong interactions between number of trees, tree
  depth, and learning rate" in cross-validation results, i.e. **tune them jointly,
  not one at a time** (p31, p35).
- Named, course-endorsed advanced boosting implementations: **XGBoost** (built-in
  regularization) and **LightGBM** ("maximize accuracy with fewer splits") (p31).
- CV captures the underfitting/overfitting trade-off for both families; "the
  computational requirements can be very significant: [...] cross-validation does it
  many times" (p25, p35).

**Interpretability**
- Ensembles are explicitly **less interpretable than a single CART** because many
  trees are combined (p37).
- Two named mitigations, both expected: **variable importance** (how much the model
  uses each feature, via total improvement from splitting on it across all trees)
  and **SHAP** (Shapley values: marginal contribution of a feature versus all
  "submodels" that omit it) (p37-38). A SHAP summary/beeswarm plot is shown as the
  standard visualization (p38).

**Business framing**
- Poll explicitly tests understanding of the random-forest-vs-boosting
  computational trade-off (parallelizable vs. sequential) (p39).
- Takeaway: "tree ensembles are less transparent than a single tree, but feature
  importance and SHAP values can help interpret predictions" -- interpretability is
  treated as a cost to be actively mitigated, not ignored, when moving to ensembles
  (p41).

---

## Lecture 7 (S07, 2026-09-30) -- Recommendation systems: collaborative filtering & stacking

**Train/test splits**
- 99/1 split -- justified by the dataset being very large (n ~ 1M), so even a 1%
  test set is a statistically meaningful holdout (p25).

**Baselines**
- "Movie average" (item-level mean rating) reported as the baseline row in every
  results table, alongside collaborative filtering and the final ensemble (p29,
  p35).

**Metrics**
- R², OSR², RMSE, MAE reported per model, same conventions as other lectures
  (p29, p35).

**Cross-validation / regularization**
- Archetype count (the model's main hyperparameter / complexity knob) is chosen via
  cross-validation at the "sweet spot" between underfitting (too few archetypes to
  capture preference diversity) and overfitting (too many; memorizes rather than
  generalizes) (p19, p23).
- **Stacking / meta-model leakage rule**: feeding a upstream model's *in-sample*
  predictions into a downstream meta-model would overstate the meta-model's
  performance; use **out-of-fold predictions** (from cross-validation) as the
  meta-model's input features instead, so the meta-model only ever sees
  out-of-sample-style inputs during its own training (p28, p33).

**Interpretability**
- Archetypes can be given a human interpretation post hoc (e.g., by genre), but
  "archetypes [are] learned from data, [their] interpretations [are] not unique"
  (p24) -- a caution against over-interpreting latent/learned representations.

**Business framing**
- Multiple business objectives beyond raw predictive error: hit-rate in
  recommended lists, catalog diversity/discovery (recommending outside the top-100
  most popular titles), and how performance varies depending on the counterfactual
  ("decoy") used to simulate user choice (p37-38).

---

## Lecture 8 (S08, 2026-10-05) -- Neural networks & deep learning intro (click-through rate)

**Train/test splits**
- 70/30 split (p13).
- **Lag/historical features carry two explicit cautions**: (1) "data leakage: they
  must always be built using past data [only]", and (2) "cold start: they do not
  allow the same prediction quality for new [entities]" with no history yet (p10).

**Baselines**
- Historical spend / "advertiser's track record" used as the baseline against which
  model-driven budget allocation is compared, alongside a "perfect hindsight" upper
  bound (p36-37).

**Metrics**
- Full **in-sample vs. out-of-sample** RMSE/MAE/R² table across every model family
  (linear, ridge, CART, random forest, gradient boosting, neural network), same
  convention as L4/L6 (p18, p38).
- "Strongest performance obtained with tree ensembles [...] limited value of
  regularization due to the limited number of features [...] limited value of
  non-linear [NN] models to capture complex patterns in the data [on this tabular
  dataset]" (p18).

**Cross-validation / regularization**
- Neural-network hyperparameters: architecture (depth, width per layer, activation
  function), regularization strategy (dropout, early stopping, loss function),
  training algorithm (batch size, learning rate, epochs) (p31).
- **"Validation, rather than cross-validation, due to the computational costs of
  deep learning"** -- an explicit, acknowledged exception to the general
  k-fold-CV recommendation, justified purely by compute cost for very expensive
  model families (p31).
- Underfitting vs. overfitting trade-off again framed as a "flat validation curve"
  with a sweet spot; "deep learning models dominating but not necessarily
  predictive trees with less tuning" (p37).

**Interpretability**
- A neural network is explicitly shown to generalize a linear/logistic model (zero
  hidden layers = equivalent linear/logistic regression) -- framing deep learning as
  a strict extension, not a wholesale replacement, of the earlier lectures' models
  (p33).

**Business framing**
- Budget-allocation decision example with a full profit table (click rate,
  clicks, cost/click, profit, "% of perfect-hindsight impact") comparing the
  historical/random policy, advertiser's own track record, and each model family
  (p36). Optimization curve of campaign profit vs. number of placements funded,
  explicitly benchmarked against "perfect hindsight" and showing a concrete "30%
  improvement from historical spend" headline (p37).

---

## Lecture 9 (S09, 2026-10-07) -- Training & hyperparameter tuning for neural networks (heart disease)

**Train/test splits**
- n=303, with an explicit train/validation split (242/61) reported in the data
  snapshot (p9).

**Baselines**
- Naive baseline for an imbalanced binary outcome: predicting the majority class
  (no disease) achieves 72.6% accuracy -- stated explicitly before any model result,
  same convention as L3 (p9).

**Metrics**
- Same loss-function framing as L3: cross-entropy for classification, SSE for
  regression (p14).

**Cross-validation / regularization**
- **Three named, explicit techniques to combat overfitting**, each tied to a
  specific failure mode (p39-41):
  1. **Dropout** -- "the network is too complex (too deep)": zero out a random
     subset of hidden-layer outputs during training (never at inference).
  2. **Early stopping** -- "the network is too well trained": stop updating weights
     once validation loss stops improving, even though training loss is still
     falling.
  3. **Regularization (L2 weight penalty)** -- "the network has too much freedom":
     add a penalty term on the squared weights to the training objective, the same
     mechanism as ridge regression in L4.
- **Standardize inputs using the training set's mean/std only**, "to avoid data
  leakage from the test set" -- same rule as L4, restated (p45).
- Hyperparameters are explicitly defined as "architecture, regularization strategy
  and training algorithm [choices], set to maximize out-of-sample performance on a
  VALIDATION set" (p48).
- Joint tuning emphasized again: "the best learning rate depends on the network
  architecture" -- tune hyperparameters together, not independently (p47).

**Interpretability**
- Not a focus of this lecture (covered in L6).

**Business framing**
- Cost-based decision rule: call a patient back for further testing if predicted
  probability of multi-vessel disease exceeds a threshold chosen from the relative
  cost of a callback ($1,500) vs. a missed multi-vessel case ($25,000) -- same
  cost-asymmetric threshold-setting logic as L3's churn discount decision (p46).

---

## Cross-lecture summary (recurring rules, by category)

| Category | The recurring rule | Lectures |
|---|---|---|
| Splits | Hold out a test set; never touch it until final evaluation | L2-L9 (every lecture) |
| Splits | Use a **temporal** split, not random, when data has a time order and leakage risk | L2, L6 |
| Splits | Stratify the split to preserve class/outcome balance | L2, L3 |
| Baselines | Always report a naive/historical baseline next to every model | L2, L3, L5, L7, L8, L9 |
| Metrics | Report in-sample AND out-of-sample metrics side by side, every model | L4, L5, L6, L8 |
| Metrics | Check **calibration** (sum of predictions vs. sum of actuals), not just per-row error | L5 |
| Metrics | Statistical significance != predictive / business value | L2, L4 |
| CV | A single validation split gives unstable hyperparameter choices; use k-fold CV | L4, L5, L6 |
| CV | Group/block CV folds to respect leakage structure (time, shared entities) | L6 |
| CV | Refit the chosen hyperparameters on the full training set before final evaluation | L4 |
| Regularization | Standardize/fit any transformer on training data only | L4, L9 |
| Regularization | Ridge/LASSO (L2/L1) penalty, or dropout/early-stopping/L2 for NNs | L4, L9 |
| Regularization | Regularized/selected coefficients carry no significance test | L4 |
| Interpretability | Trees/linear models: coefficients, thresholds, split order | L2, L3, L5 |
| Interpretability | Ensembles: variable importance + SHAP, because the model itself is opaque | L6 |
| Extrapolation | **Trees cannot predict outside the training data's range** | L5 |
| Business framing | Convert the model's output into a decision with a cost/profit calculation, benchmarked against a baseline or hindsight | L2, L3, L5, L7, L8, L9 |
| Business framing | Choose a decision threshold to maximize business profit, not a default (e.g. 0.5) | L3, L9 |
