import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import training_functions as tf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV

from sklearn import set_config

set_config(enable_metadata_routing=True)


def main():
    """Execute the complete ML pipeline for TRESH-GACC threshold evaluation.

    This function performs a comprehensive machine learning experiment to evaluate
    different threshold values for the TRESH-GACC metric. For each threshold,
    it trains an SVM classifier with grid search optimization and evaluates
    performance on both validation and test sets.

    Pipeline Steps:
        1. Load synthetic MNIST patient data
        2. Split data using stratified group k-fold (preserving patient groups)
        3. For each threshold value from 0.01 to 0.95:
           - Create pipeline: StandardScaler → PCA(50) → SVM(RBF kernel)
           - Perform grid search over C and gamma parameters
           - Train on training set with 5-fold group cross-validation
           - Evaluate on held-out test set
           - Save results and predictions

    Hyperparameter Grid:
        - C: 15 values logarithmically spaced from 10^-1 to 10^9
        - gamma: 15 values logarithmically spaced from 10^-12 to 10^0
        - Total combinations: 225 parameter sets per threshold

    Outputs:
        For each threshold value:
        - models/results_cutoff_{threshold:.2f}.pkl: Trained GridSearchCV object
        - models/results_cutoff_{threshold:.2f}.csv: Test predictions and metadata

    Data Requirements:
        - Input: 'data/synthetic_mnist_data.pkl' (created by create_data.py)
        - Output directories: 'models/' must exist

    Performance Metrics:
        - Cross-validation: TRESH-GACC with specified threshold
        - Test evaluation: TRESH-GACC with same threshold

    Example Output:
        Evaluating with cutoff=0.01...
            Best parameters: {'svc__C': ..., 'svc__gamma': ...}
            Best cross-validation score: ...
            Test set TRESH-GACC: ...
        Evaluating with cutoff=0.03...
        ...

    Note:
        - Uses all available CPU cores (n_jobs=-1) for parallel processing
        - Maintains patient-level groups throughout train/test splits
        - Evaluates 40 different threshold values for comprehensive analysis
    """
    df = pd.read_pickle(r"data/synthetic_mnist_data.pkl")

    cv_holdout = StratifiedGroupKFold(n_splits=5, shuffle=True)
    train_idx, test_idx = next(cv_holdout.split(df, df["label"], df["patient_id"]))

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=25)),
            ("svc", SVC(kernel="rbf")),
        ]
    )
    param_grid = {
        "svc__C": np.logspace(-2, 6, 15),
        "svc__gamma": np.logspace(-10, 0, 15),
    }
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True)
    thresholds = [0.01, 0.05, 0.1, 0.5, 0.75]

    for thresh in thresholds:
        print(f"Evaluating with cutoff={thresh:.2f}...")
        scorer = tf.tresh_gacc_scorer(cutoff=thresh)
        grid_search = GridSearchCV(
            pipe, param_grid, cv=cv, scoring=scorer, n_jobs=-1, refit=True
        )

        X_train = np.stack(df.loc[train_idx, "px_0":"px_783"].values)
        y_train = df.loc[train_idx, "label"].values
        groups_train = df.loc[train_idx, "patient_id"].values
        X_test = np.stack(df.loc[test_idx, "px_0":"px_783"].values)
        y_test = df.loc[test_idx, "label"].values
        groups_test = df.loc[test_idx, "patient_id"].values
        grid_search.fit(X_train, y_train, groups=groups_train)
        print(f"\tBest parameters: {grid_search.best_params_}")
        print(f"\tBest cross-validation score: {grid_search.best_score_:.4f}")
        classifier = grid_search.best_estimator_
        y_pred = classifier.predict(X_test)
        test_score = tf.tresh_gacc(y_test, y_pred, groups=groups_test, cutoff=thresh)
        print(f"\tTest set TRESH-GACC: {test_score:.4f}")

        with open(Path(r"models") / f"results_cutoff_{thresh:.2f}.pkl", "wb") as f:
            pickle.dump(grid_search, f)

        df.loc[test_idx, "y_pred"] = y_pred
        df.loc[test_idx, ["patient_id", "roi", "label", "y_pred"]].to_csv(
            Path(r"models") / f"results_cutoff_{thresh:.2f}.csv", index=False
        )


if __name__ == "__main__":
    main()
