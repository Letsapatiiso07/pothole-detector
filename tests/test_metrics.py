import pytest

from utils import binary_metrics, bootstrap_ci


def test_binary_metrics_known_counts():
    # 3 TP, 1 FP, 2 FN, 4 TN
    y_true = [1, 1, 1, 0, 1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    m = binary_metrics(y_true, y_pred)
    assert (m['tp'], m['fp'], m['fn'], m['tn']) == (3, 1, 2, 4)
    assert m['accuracy'] == pytest.approx(0.7)
    assert m['precision'] == pytest.approx(0.75)
    assert m['recall'] == pytest.approx(0.6)
    assert m['f1'] == pytest.approx(2 * 0.75 * 0.6 / 1.35)


def test_binary_metrics_no_positive_predictions():
    m = binary_metrics([1, 0, 1], [0, 0, 0])
    assert m['precision'] == 0.0 and m['recall'] == 0.0 and m['f1'] == 0.0


def test_bootstrap_ci_contains_point_estimate():
    y_true = [1, 0] * 50
    y_pred = [1, 0] * 40 + [0, 1] * 10
    lo, hi = bootstrap_ci(y_true, y_pred, 'accuracy', n=500)
    assert lo <= 0.8 <= hi
