"""Tests for seldonian.datasets. These need the optional `datasets` extra
(shap + tempeh) and network access; they skip when unavailable."""

import numpy as np
import pytest

shap = pytest.importorskip("shap")

from seldonian.datasets import AdultDataset  # noqa: E402


class TestAdultDataset:
    def test_get_data_contract(self):
        ds = AdultDataset()
        X, X_test, y, y_test, A, A_idx = ds.get_data()
        assert X.shape[0] == y.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        assert X.shape[1] == X_test.shape[1]
        assert 0 <= A_idx < X.shape[1]
        assert isinstance(A, str)
        # sensitive column is binary
        assert set(np.unique(X[:, A_idx])) <= {0.0, 1.0}


@pytest.mark.skipif(
    pytest.importorskip("tempeh", reason="tempeh not installed") is None,
    reason="tempeh not installed")
class TestLawschoolDataset:
    def test_get_data_contract(self):
        from seldonian.datasets import LawschoolDataset

        ds = LawschoolDataset(n=300)
        X, X_test, y, y_test, A, A_idx = ds.get_data()
        assert X.shape[0] == y.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        assert 0 <= A_idx < X.shape[1]
