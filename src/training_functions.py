from sklearn.metrics import make_scorer, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def tresh_gacc_scorer(cutoff):
    """Create a scikit-learn scorer for TRESH-GACC (thresholded group accuracy).

    This function creates a scorer that can be used with scikit-learn's model
    evaluation tools (e.g., GridSearchCV). The scorer calculates group-level
    accuracy based on a threshold for positive predictions within each group.

    Args:
        cutoff (float): The threshold for determining positive group predictions.
                       Groups with >= cutoff fraction of positive predictions
                       are classified as positive.

    Returns:
        sklearn.metrics.make_scorer: A scorer object that can be used with
                                    scikit-learn evaluation methods. Requires
                                    groups parameter to be passed for evaluation.

    Example:
        >>> scorer = tresh_gacc_scorer(cutoff=0.25)
        >>> # Use with GridSearchCV
        >>> grid_search = GridSearchCV(model, param_grid, scoring=scorer)
    """
    return make_scorer(tresh_gacc, cutoff=cutoff).set_score_request(groups=True)


def tresh_gacc(y_true, y_pred, *, groups=None, cutoff=0.25):
    """Calculate thresholded group accuracy (TRESH-GACC) metric.

    TRESH-GACC evaluates model performance at the group level rather than
    individual predictions. Each group is classified as positive if the
    fraction of positive predictions within that group meets or exceeds
    the specified cutoff threshold.

    Args:
        y_true (array-like): True binary labels for individual samples.
        y_pred (array-like): Predicted binary labels for individual samples.
        groups (array-like, optional): Group identifiers for each sample.
                                     Samples with the same identifier belong
                                     to the same group. Defaults to None.
        cutoff (float, optional): Threshold for group-level classification.
                                Groups with >= cutoff fraction of positive
                                predictions are classified as positive.
                                Defaults to 0.25.

    Returns:
        float: Group-level accuracy score between 0 and 1, where 1 is perfect
               group-level accuracy.

    Example:
        >>> # Patient 1: all ROIs negative, Patient 2: all ROIs positive
        >>> y_true  = [0, 0, 1, 1]
        >>> y_pred  = [0, 0, 1, 1]  # perfect predictions
        >>> groups  = [1, 1, 2, 2]
        >>> tresh_gacc(y_true, y_pred, groups=groups, cutoff=0.5)
        1.0
        >>> y_pred2 = [1, 1, 0, 0]  # inverted predictions
        >>> tresh_gacc(y_true, y_pred2, groups=groups, cutoff=0.5)
        0.0
    """
    yg_true = group_predict(y_true, groups=groups, cutoff=cutoff)
    yg_predict = group_predict(y_pred, groups=groups, cutoff=cutoff)

    return accuracy_score(yg_true, yg_predict)


def group_predict(y_pred, groups=None, cutoff=0.25, pos_label=1, neg_label=0):
    """Generate group-level predictions based on individual predictions and threshold.

    This function aggregates individual predictions within each group and generates
    a single prediction per group based on whether the fraction of positive
    predictions meets or exceeds the specified cutoff threshold.

    Args:
        y_pred (array-like): Individual binary predictions for each sample.
        groups (array-like, optional): Group identifiers for each sample.
                                     Samples with the same identifier belong
                                     to the same group. Defaults to None.
        cutoff (float, optional): Threshold for group-level classification.
                                Groups with >= cutoff fraction of positive
                                predictions are classified as positive.
                                Defaults to 0.25.
        pos_label (int, optional): Label for positive class. Defaults to 1.
        neg_label (int, optional): Label for negative class. Defaults to 0.

    Returns:
        numpy.ndarray: Array of group-level predictions, one per unique group.
                      Array length equals the number of unique groups.

    Example:
        >>> y_pred = [1, 1, 0, 0, 1, 1]
        >>> groups = ["A", "A", "A", "B", "B", "B"]
        >>> group_predict(y_pred, groups, cutoff=0.5)
        array([1, 1])  # Group A: 2/3=0.67≥0.5→1, Group B: 2/3=0.67≥0.5→1
    """
    groups = np.array(groups)
    y_pred = np.array(y_pred)
    identifier = np.array(list(set(groups)))
    result = []
    for i in identifier:
        sel = groups == i
        positive = np.where(y_pred[sel] == pos_label, 1, 0)
        if np.mean(positive) >= cutoff:
            result.append(pos_label)
        else:
            result.append(neg_label)
    return np.array(result)


def plot_grid_search_results(
    df, title="Grid Search Results: Mean Test Score by C and Gamma Parameters", ax=None
):
    """Create a heatmap visualization of grid search results.

    This function generates a heatmap showing the relationship between SVM
    hyperparameters (C and gamma) and model performance (mean test score).
    The axes labels are formatted in scientific notation for better readability.

    Args:
        df (pandas.DataFrame): DataFrame containing grid search results with
                              columns 'mean_test_score', 'param_svc__gamma',
                              and 'param_svc__C'.
        title (str, optional): Title for the heatmap plot. Defaults to
                              "Grid Search Results: Mean Test Score by C and Gamma Parameters".
        ax (matplotlib.axes.Axes, optional): Existing axes object to plot on.
                                           If None, creates new figure and axes.
                                           Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axes object containing the heatmap plot.

    Example:
        >>> ax = plot_grid_search_results(df, title="Custom Title")
        >>> # ax contains the heatmap with scientific notation tick labels
    """

    # Pivot the data to create a matrix for heatmap
    heatmap_data = df.pivot_table(
        values="mean_test_score", index="param_svc__gamma", columns="param_svc__C"
    )

    # Create the heatmap
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": "Mean Test Score"},
    )
    ax.set_xticklabels([f"{float(i.get_text()):.2e}" for i in ax.get_xticklabels()])
    ax.set_yticklabels([f"{float(i.get_text()):.2e}" for i in ax.get_yticklabels()])
    ax.set_title(title)
    ax.set_xlabel("C Parameter")
    ax.set_ylabel("Gamma Parameter")

    return ax
