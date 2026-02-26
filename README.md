# TRESH-GACC

A scikit-learn compatible scorer for **group-level accuracy** in Multiple Instance Learning (MIL) settings.

Standard scikit-learn metrics evaluate predictions at the sample level. TRESH-GACC shifts evaluation to the **group (bag) level**: instance predictions within a group are aggregated using a configurable threshold, and accuracy is then computed between the resulting group-level labels. This makes it a natural fit for MIL workflows where labels are known only at the group level (e.g. patients, slides, scenes) but the model operates on individual instances (e.g. ROIs, patches, frames).

---

## Scorer Usage

TRESH-GACC integrates with any scikit-learn estimator or hyperparameter search that accepts a `scoring` argument. Because it requires group information at evaluation time, it uses scikit-learn's **metadata routing** API.

```python
from sklearn import set_config
set_config(enable_metadata_routing=True)

from training_functions import tresh_gacc_scorer

scorer = tresh_gacc_scorer(cutoff=0.25)
```

### With `cross_val_score`

```python
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold

scores = cross_val_score(
    estimator,
    X, y,
    cv=StratifiedGroupKFold(n_splits=5),
    scoring=scorer,
    params={"groups": patient_ids},
)
```

### With `GridSearchCV`

```python
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

grid_search = GridSearchCV(
    estimator,
    param_grid,
    cv=StratifiedGroupKFold(n_splits=5),
    scoring=scorer,
)
grid_search.fit(X_train, y_train, groups=patient_ids)
```

The `groups` array is passed once at `.fit()` time and automatically forwarded to the scorer at each cross-validation split.

---

## How It Works

For a given `cutoff` threshold τ, each group is classified as positive if the fraction of its instances predicted positive is ≥ τ:

```
group_pred_i = 1  if  (# positive predictions in group i) / (# instances in group i) >= τ
             = 0  otherwise

TRESH-GACC = accuracy(group_true, group_pred)
```

The same aggregation is applied to the true labels, so a group is considered truly positive only if ≥ τ of its instances carry a positive label.

| Cutoff | Bag-labelling rule |
|--------|--------------------|
| ~0 (very small) | One positive instance suffices — equivalent to standard MIL assumption |
| 0.5 | Majority vote across instances |
| 1.0 | All instances must be positive |

Sweeping the cutoff lets you explore the full spectrum of bag-labelling assumptions without changing the underlying model.

---

## Multiple Instance Learning Context

TRESH-GACC is designed for the MIL setting (Dietterich et al., 1997), in which:

- **Bags** (groups) are labelled, but individual **instances** within a bag are not
- A model trained on instances is evaluated on whether it correctly recovers the bag-level label

The adjustable threshold generalises the classical "positive-if-any" MIL assumption, adding robustness in domains where a single noisy instance should not determine the bag label — common in histopathology, radiology, and video classification.

TRESH-GACC works with **any** scikit-learn-compatible instance-level classifier; it is purely a scoring function and imposes no constraints on the model architecture.

---

## Demonstration

This repository includes a full demonstration on a synthetic medical imaging dataset built from MNIST digits:

- **60 patients**, each with 3–19 ROIs
- **Negative patients** contain only digit '9' images; **positive patients** contain at least one digit '1' among '9's
- An **SVM pipeline** (StandardScaler → PCA → RBF-SVM) is trained at the instance level and evaluated with TRESH-GACC across 40 threshold values (0.01 to 0.95)

### TRESH-GACC vs Cutoff Threshold

![TRESH-GACC vs cutoff](figures/thresholded_gacc_vs_cutoff.png)

### Grid Search Results

TRESH-GACC used as the scoring function inside `GridSearchCV`, sweeping C and gamma of the RBF-SVM. The optimal hyperparameters shift with the choice of cutoff, illustrating that threshold selection is a meaningful modelling decision.

**Cutoff = 0.05**

![Grid search cutoff 0.05](figures/grid_search_cutoff_0.05.png)

**Cutoff = 0.50**

![Grid search cutoff 0.50](figures/grid_search_cutoff_0.50.png)

### Running the Demo

```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

python src/create_data.py   # generate synthetic dataset
python src/train.py         # run grid search across all thresholds
# open notebooks/visualize_gridsearch.ipynb for figures
```

---

## Project Structure

```
thresh_gacc/
├── data/
│   └── synthetic_mnist_data.pkl
├── figures/
│   ├── thresholded_gacc_vs_cutoff.png
│   └── grid_search_cutoff_*.png
├── models/
│   ├── results_cutoff_*.pkl
│   └── results_cutoff_*.csv
├── notebooks/
│   ├── develop_training.ipynb
│   └── visualize_gridsearch.ipynb
├── src/
│   ├── __init__.py
│   ├── create_data.py         # synthetic dataset generation
│   ├── train.py               # demo training pipeline
│   └── training_functions.py  # scorer, metric, and plotting
├── .gitignore
├── README.md
└── requirements.txt
```

---

## API

#### `tresh_gacc_scorer(cutoff)` → sklearn scorer
Drop-in scorer for use with any scikit-learn evaluation tool. Requires `enable_metadata_routing=True`.

#### `tresh_gacc(y_true, y_pred, *, groups, cutoff)` → float
Compute TRESH-GACC directly. Useful for post-hoc evaluation outside of a pipeline.

#### `group_predict(y_pred, groups, cutoff)` → ndarray
Aggregate instance predictions to group-level predictions given a threshold.

#### `plot_grid_search_results(df, title, ax)` → Axes
Heatmap of `GridSearchCV.cv_results_` with C and gamma on the axes.

---

## Dependencies

- scikit-learn >= 1.4 (metadata routing API)
- numpy, pandas, matplotlib, seaborn

Full list in `requirements.txt`.

---

## References

- Dietterich, T. G., Lathrop, R. H., & Lozano-Perez, T. (1997). Solving the multiple instance problem with axis-parallel rectangles. *Artificial Intelligence*, 89(1-2), 31-71.
- Ilse, M., Tomczak, J., & Welling, M. (2018). Attention-based deep multiple instance learning. *ICML*. https://arxiv.org/abs/1802.04712
- Scikit-learn metadata routing: https://scikit-learn.org/stable/metadata_routing.html

