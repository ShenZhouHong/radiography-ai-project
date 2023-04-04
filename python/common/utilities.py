#!/usr/bin/env python3

def calculate_mean_metrics(metrics_list: list[dict[str, float]]) -> dict[str, list[float]]:
    """
    Calculates the mean value of each metric across multiple folds of a K-fold cross validation for a machine learning model.

    Args:
        metrics_list (list[dict[str, float]]): A list of dictionaries containing the metrics for each fold of the
            K-fold cross validation. Each dictionary should have keys for each metric and values for each epoch.

    Returns:
        dict[str, list[float]]: A dictionary containing the mean value of each metric across all folds of the
            K-fold cross validation. The keys of the dictionary are the metric names, and the values are lists
            containing the mean value of the metric for each epoch.
    """

    # Initialise aggregate metrics
    aggregate_metrics: dict[str, list[float]] = {}
    for fold in metrics_list:
        for metric in fold.keys():
            if metric not in aggregate_metrics:
                aggregate_metrics[metric] = []

    # Calculate the average metric per epoch for every fold
    number_of_folds: int = len(metrics_list)
    for metric in aggregate_metrics.keys():
        number_of_epochs: int = len(metrics_list[0][metric])
        for epoch in range(number_of_epochs):
            # A list of every value for that given metric in this epoch across folds
            values_per_epoch: list[float] = [x[metric][epoch] for x in metrics_list]
            mean_per_epoch  : float = sum(values_per_epoch) / number_of_folds
            aggregate_metrics[metric].append(mean_per_epoch)

    return aggregate_metrics

def calculate_avg_max(k_fold_metrics: list[dict[str, float]], metric_name: str = "val_auc") -> float:
    """
    Calculates the maximum average metric value across all folds of a k-fold cross-validation experiment.

    Args:
        k_fold_metrics (list[dict[str, float]]): A list of dictionaries containing metric values for each fold.
            Each dictionary should contain the same set of keys representing the names of the metrics,
            and the corresponding values representing the metric values for the corresponding fold.
        metric_name (str, optional): The name of the metric for which the maximum average value is to be calculated.
            Defaults to "val_auc".
    Returns:
        float: The maximum average metric value across all folds.

    """
    avg_metrics: dict[str, list[float]] = calculate_mean_metrics(k_fold_metrics)
    max_avg_metric: float = max(avg_metrics[metric_name])

    return max_avg_metric