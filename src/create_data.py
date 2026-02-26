from sklearn.datasets import fetch_openml
import numpy as np
from pathlib import Path
import pandas as pd
import os


def create_synthetic_data():
    """Create synthetic patient-level dataset from MNIST digits.

    This function generates a synthetic medical imaging dataset where each
    patient has multiple ROIs (regions of interest) represented by MNIST
    digit images. Patients are assigned binary labels based on whether
    they have any digit '1' images among their ROIs.

    Dataset Structure:
        - 60 patients total (30 with label 0, 30 with label 1)
        - Each patient has 3-19 ROIs randomly assigned
        - Label 0 patients: Only contain digit '9' images
        - Label 1 patients: Contain at least one digit '1' image, rest are '9's
        - Each ROI is represented by 784 pixel values (28x28 flattened)

    Saves:
        pandas.DataFrame: Saves dataset to 'data/synthetic_mnist_data.pkl'
                         with columns:
                         - patient_id: Patient identifier (0-59)
                         - roi: ROI index within patient (0-based)
                         - label: Binary patient-level label (0 or 1)
                         - px_0 to px_783: Pixel values for the ROI image

    Example Output Structure:
        >>> df = pd.read_pickle('data/synthetic_mnist_data.pkl')
        >>> print(df.shape)
        # (~700-900 rows depending on random ROI counts, 787 columns)
        # Columns: patient_id, roi, label, px_0 ... px_783
        >>> print(df.groupby('patient_id')['label'].first().value_counts().to_dict())
        {0: 30, 1: 30}

    Note:
        Uses random seed 42 for reproducibility.
        Requires 'data' directory to exist for saving output.
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    x_all = mnist.data.astype(np.uint8)  # shape (70000, 784) — already flat!
    y_all = mnist.target.astype(int)  # shape (70000,)

    idx_9 = np.where(y_all == 9)[0]
    idx_1 = np.where(y_all == 1)[0]

    np.random.seed(42)

    N_PATIENTS = 60
    # Half get label 0 (only 9s), half get label 1 (9s + at least one 1)
    labels = [0] * 30 + [1] * 30
    np.random.shuffle(labels)

    rows = []

    for patient_id, label in enumerate(labels):
        n_rois = np.random.randint(3, 20)  # 3 to 19 inclusive

        if label == 0:
            # Only 9s
            chosen_idx = np.random.choice(idx_9, size=n_rois, replace=True)
        else:
            # At least one 1, rest are 9s
            n_ones = np.random.randint(1, min(10, n_rois))
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
