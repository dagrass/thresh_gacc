from sklearn.datasets import fetch_openml
import numpy as np
from pathlib import Path
import pandas as pd
import os


def create_synthetic_data():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    x_all = mnist.data.astype(np.uint8)  # shape (70000, 784) — already flat!
    y_all = mnist.target.astype(int)  # shape (70000,)

    idx_9 = np.where(y_all == 9)[0]
    idx_1 = np.where(y_all == 1)[0]

    np.random.seed(42)

    N_PATIENTS = 80
    # Half get label 0 (only 9s), half get label 1 (9s + at least one 1)
    labels = [0] * 40 + [1] * 40
    np.random.shuffle(labels)

    rows = []

    for patient_id, label in enumerate(labels):
        n_rois = np.random.randint(3, 16)  # 3 to 15 inclusive

        if label == 0:
            # Only 9s
            chosen_idx = np.random.choice(idx_9, size=n_rois, replace=True)
        else:
            # At least one 1, rest are 9s
            n_ones = np.random.randint(1, min(5, n_rois))
            n_nines = n_rois - n_ones
            chosen_idx = np.concatenate(
                [
                    np.random.choice(idx_1, size=n_ones, replace=True),
                    np.random.choice(idx_9, size=n_nines, replace=True),
                ]
            )
            np.random.shuffle(chosen_idx)

        for roi_idx, img_idx in enumerate(chosen_idx):
            vec = x_all[img_idx]
            row = {
                "patient_id": patient_id,
                "roi": roi_idx,
                "label": label,
                **{f"px_{i}": vec[i] for i in range(784)},
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    print(df.shape)
    print(df[["patient_id", "roi", "label"]].head(20))
    print(
        f"\nLabel distribution: {df.groupby('patient_id')['label'].first().value_counts().to_dict()}"
    )
    output_path = Path(r"data") / "synthetic_mnist_data.pkl"
    df.to_pickle(output_path)
    print(f"\nData saved to {output_path}")


if __name__ == "__main__":
    create_synthetic_data()
