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
    df = pd.read_pickle(r"data/synthetic_mnist_data.pkl")

    cv_holdout = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(cv_holdout.split(df, df["label"], df["patient_id"]))

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=50)),
            ("svc", SVC(kernel="rbf")),
        ]
    )
    param_grid = {
        "svc__C": np.logspace(-5, 8, 20),
        "svc__gamma": np.logspace(-7, -5, 20),
    }
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    thresholds = np.linspace(0.05, 0.95, 20)

    for thresh in thresholds:
        print(f"Evaluating with cutoff={thresh:.2f}...")
        scorer = tf.tresh_gacc_scorer(cutoff=thresh)
        grid_search = GridSearchCV(
            pipe, param_grid, cv=cv, scoring=scorer, n_jobs=1, refit=True
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
        df.loc[test_idx].to_csv(
            Path(r"models") / f"results_cutoff_{thresh:.2f}.csv", index=False
        )


if __name__ == "__main__":
    main()
