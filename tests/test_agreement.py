"""Tests for agreement.py (T6b-4) with hand-computed toy cases."""
import pytest

from code.eval.agreement import (
    binarise_faithfulness,
    compute_agreement,
    exact_agreement,
    f1_for_class,
    make_confusion,
    quadratic_kappa,
    unweighted_kappa,
)


# ── binarise_faithfulness ────────────────────────────────────────────

class TestBinarise:
    def test_perfect(self):
        assert binarise_faithfulness(1.0) == "N"

    def test_imperfect(self):
        assert binarise_faithfulness(0.75) == "Y"

    def test_zero(self):
        assert binarise_faithfulness(0.0) == "Y"

    def test_none(self):
        assert binarise_faithfulness(None) is None


# ── exact_agreement ──────────────────────────────────────────────────

class TestExactAgreement:
    def test_perfect(self):
        assert exact_agreement([1, 0.5, 0], [1, 0.5, 0]) == 1.0

    def test_none(self):
        assert exact_agreement([1, 0, 1], [0, 1, 0]) == 0.0

    def test_partial(self):
        assert exact_agreement([1, 1, 0, 0], [1, 0, 0, 1]) == 0.5

    def test_empty(self):
        assert exact_agreement([], []) == 0.0


# ── f1_for_class ─────────────────────────────────────────────────────

class TestF1ForClass:
    def test_perfect_recall_precision(self):
        assert f1_for_class([1, 1, 0], [1, 1, 0], pos_label=1) == 1.0

    def test_no_positive_predictions(self):
        assert f1_for_class([1, 1, 0], [0, 0, 0], pos_label=1) == 0.0

    def test_partial(self):
        # true=[1,1,0,0], pred=[1,0,1,0]
        # TP=1, FP=1, FN=1 → P=0.5, R=0.5, F1=0.5
        assert f1_for_class([1, 1, 0, 0], [1, 0, 1, 0], pos_label=1) == 0.5


# ── quadratic_kappa with hand-computed value ─────────────────────────

class TestQuadraticKappa:
    def test_perfect(self):
        y1 = [0, 0.5, 1, 1, 0]
        y2 = [0, 0.5, 1, 1, 0]
        assert quadratic_kappa(y1, y2) == 1.0

    def test_hand_computed(self):
        """Hand-computed quadratic-weighted kappa on a small toy case.

        Ratings on [0, 0.5, 1]:
          hand  = [1,   1,   0.5, 0,   1  ]
          judge = [1,   0.5, 0.5, 0,   0.5]

        Observed matrix (rows=hand, cols=judge), labels=[0, 0.5, 1]:
              0   0.5   1
          0 [ 1,  0,    0 ]
        0.5 [ 0,  1,    0 ]
          1 [ 0,  2,    1 ]

        n = 5, k = 3, max_dist = 2
        Quadratic weights w_ij = 1 - ((i-j)/(k-1))^2 for levels [0,1,2]:
          w = [[1, 0.75, 0],
               [0.75, 1, 0.75],
               [0, 0.75, 1]]

        Weighted observed = sum(O_ij * w_ij):
          = 1*1 + 1*1 + 2*0.75 + 1*1 = 4.5
        po = 4.5 / 5 = 0.9

        Marginals: hand = [1, 1, 3], judge = [1, 3, 1]
        Expected E_ij = hand_i * judge_j / n:
          E = [[0.2, 0.6, 0.2],
               [0.2, 0.6, 0.2],
               [0.6, 1.8, 0.6]]
        Weighted expected = sum(E_ij * w_ij):
          = 0.2*1 + 0.6*0.75 + 0.2*0
          + 0.2*0.75 + 0.6*1 + 0.2*0.75
          + 0.6*0 + 1.8*0.75 + 0.6*1
          = 0.2 + 0.45 + 0
          + 0.15 + 0.6 + 0.15
          + 0 + 1.35 + 0.6
          = 3.5
        pe = 3.5 / 5 = 0.7

        kappa = (po - pe) / (1 - pe) = (0.9 - 0.7) / (1 - 0.7) = 0.2 / 0.3 = 2/3
        """
        hand = [1, 1, 0.5, 0, 1]
        judge = [1, 0.5, 0.5, 0, 0.5]
        k = quadratic_kappa(hand, judge, labels=[0, 0.5, 1])
        assert abs(k - 2 / 3) < 1e-4

    def test_complete_disagreement(self):
        """All off-diagonal: kappa should be negative."""
        hand = [0, 0, 1, 1]
        judge = [1, 1, 0, 0]
        k = quadratic_kappa(hand, judge, labels=[0, 0.5, 1])
        assert k < 0


# ── unweighted_kappa with hand-computed value ────────────────────────

class TestUnweightedKappa:
    def test_perfect(self):
        y1 = ["Y", "N", "Y", "N"]
        y2 = ["Y", "N", "Y", "N"]
        assert unweighted_kappa(y1, y2) == 1.0

    def test_hand_computed(self):
        """Hand-computed unweighted kappa on a toy binary case.

        hand  = [Y, Y, N, N, Y]
        judge = [Y, N, N, Y, Y]

        Confusion (rows=hand, cols=judge):
              N  Y
          N [ 1, 1 ]
          Y [ 1, 2 ]

        po = (1 + 2) / 5 = 0.6
        hand_N=2, hand_Y=3, judge_N=2, judge_Y=3
        pe = (2*2 + 3*3) / 25 = (4+9)/25 = 13/25 = 0.52
        kappa = (0.6 - 0.52) / (1 - 0.52) = 0.08 / 0.48 = 1/6
        """
        hand = ["Y", "Y", "N", "N", "Y"]
        judge = ["Y", "N", "N", "Y", "Y"]
        k = unweighted_kappa(hand, judge, labels=["N", "Y"])
        assert abs(k - 1 / 6) < 1e-4


# ── confusion matrix ────────────────────────────────────────────────

class TestConfusionMatrix:
    def test_binary(self):
        cm = make_confusion(["Y", "Y", "N", "N"], ["Y", "N", "N", "Y"],
                            labels=["N", "Y"])
        # rows=true, cols=pred
        assert cm == [[1, 1], [1, 1]]

    def test_ordinal(self):
        cm = make_confusion([0, 0.5, 1], [0, 0.5, 1], labels=[0, 0.5, 1])
        assert cm == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


# ── compute_agreement integration ───────────────────────────────────

class TestComputeAgreement:
    def _pairs(self):
        return [
            {"accuracy_hand": 1, "accuracy_judge": 1,
             "faithfulness_hand": "N", "faithfulness_judge": 1.0},
            {"accuracy_hand": 1, "accuracy_judge": 0.5,
             "faithfulness_hand": "Y", "faithfulness_judge": 0.75},
            {"accuracy_hand": 0.5, "accuracy_judge": 0.5,
             "faithfulness_hand": "N", "faithfulness_judge": 1.0},
            {"accuracy_hand": 0, "accuracy_judge": 0,
             "faithfulness_hand": "Y", "faithfulness_judge": 0.5},
            {"accuracy_hand": 1, "accuracy_judge": 0.5,
             "faithfulness_hand": "Y", "faithfulness_judge": 1.0},
        ]

    def test_returns_both_scores(self):
        result = compute_agreement(self._pairs(), threshold=0.60)
        assert "accuracy" in result
        assert "faithfulness" in result

    def test_accuracy_has_kappa(self):
        result = compute_agreement(self._pairs(), threshold=0.60)
        assert result["accuracy"]["kappa"] is not None
        assert isinstance(result["accuracy"]["kappa"], float)

    def test_faithfulness_has_kappa(self):
        result = compute_agreement(self._pairs(), threshold=0.60)
        assert result["faithfulness"]["kappa"] is not None

    def test_unvalidated_flag(self):
        """Kappa below threshold → unvalidated=True."""
        # Use pairs with poor agreement
        pairs = [
            {"accuracy_hand": 1, "accuracy_judge": 0,
             "faithfulness_hand": "Y", "faithfulness_judge": 1.0},
            {"accuracy_hand": 0, "accuracy_judge": 1,
             "faithfulness_hand": "N", "faithfulness_judge": 0.0},
            {"accuracy_hand": 1, "accuracy_judge": 0,
             "faithfulness_hand": "Y", "faithfulness_judge": 1.0},
            {"accuracy_hand": 0, "accuracy_judge": 1,
             "faithfulness_hand": "N", "faithfulness_judge": 0.0},
        ]
        result = compute_agreement(pairs, threshold=0.60)
        # Complete disagreement → kappa < 0 < 0.60
        assert result["accuracy"]["unvalidated"] is True
        assert result["faithfulness"]["unvalidated"] is True

    def test_null_judge_excluded(self):
        pairs = [
            {"accuracy_hand": 1, "accuracy_judge": None,
             "faithfulness_hand": "Y", "faithfulness_judge": None},
            {"accuracy_hand": 1, "accuracy_judge": 1,
             "faithfulness_hand": "N", "faithfulness_judge": 1.0},
            {"accuracy_hand": 0, "accuracy_judge": 0,
             "faithfulness_hand": "Y", "faithfulness_judge": 0.5},
        ]
        result = compute_agreement(pairs, threshold=0.60)
        assert result["accuracy"]["null_excluded"] == 1
        assert result["accuracy"]["n"] == 2
        assert result["faithfulness"]["null_excluded"] == 1
        assert result["faithfulness"]["n"] == 2

    def test_confusion_matrix_present(self):
        result = compute_agreement(self._pairs(), threshold=0.60)
        assert "confusion_matrix" in result["accuracy"]
        assert "confusion_matrix" in result["faithfulness"]

    def test_f1_present(self):
        result = compute_agreement(self._pairs(), threshold=0.60)
        assert "f1_accuracy_1" in result["accuracy"]

    def test_too_few_pairs(self):
        """Fewer than 2 valid pairs → kappa=None, unvalidated=True."""
        pairs = [
            {"accuracy_hand": 1, "accuracy_judge": 1,
             "faithfulness_hand": "N", "faithfulness_judge": 1.0},
        ]
        result = compute_agreement(pairs, threshold=0.60)
        assert result["accuracy"]["kappa"] is None
        assert result["accuracy"]["unvalidated"] is True
