import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from svm_grid import MidpointNormalize
import pytest

class TestSVMGridSearch:

    def test_iris_number_of_samples(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        assert X.shape[0] == 150
        assert len(y) == 150


    def test_iris_number_of_features(self):
        iris = load_iris()
        X = iris.data
        assert X.shape[1] == 4


    def test_iris_number_of_classes(self):
        iris = load_iris()
        y = iris.target
        assert len(np.unique(y)) == 3  

    def test_scaler_standardization(self):
        iris = load_iris()
        X = iris.data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-7)

        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-7)

    def test_param_ranges(self):
        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)

        assert len(C_range) == 13
        assert len(gamma_range) == 13
        assert np.all(C_range > 0)
        assert np.all(gamma_range > 0)


    def test_param_grid_keys(self):
        param_grid = {
            "C": np.logspace(-2, 10, 13),
            "gamma": np.logspace(-9, 3, 13),
        }

        assert set(param_grid.keys()) == {"C", "gamma"}

    def test_cv_configuration(self):
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

        assert cv.n_splits == 5
        assert cv.test_size == 0.2
        assert cv.random_state == 42

    def test_grid_search_runs(self):
        iris = load_iris()
        X, y = iris.data, iris.target

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)
        param_grid = {"C": C_range, "gamma": gamma_range}
        cv = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=0)

        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        grid.fit(X_scaled, y)

        assert hasattr(grid, "best_params_")
        assert grid.best_score_ > 0
        assert "mean_test_score" in grid.cv_results_

    def test_midpoint_normalize_basic(self):
        norm = MidpointNormalize(vmin=0, vmax=10, midpoint=5)
        result = norm([0, 5, 10])

        expected = np.array([0.0, 0.5, 1.0])
        assert np.allclose(result.data, expected)


    def test_midpoint_normalize_masking(self):
        norm = MidpointNormalize(vmin=0, vmax=10, midpoint=5)
        result = norm([np.nan])

        assert isinstance(result, np.ma.MaskedArray)
        assert np.isnan(result.data[0])
        assert result.mask == False

if __name__ == "__main__":
    pytest.main([__file__, "-v"])