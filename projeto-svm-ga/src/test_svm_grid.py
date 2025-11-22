import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC


# Teste para garantir que o iris carrega os valores padrão corretamente
def test_iris_dataset():
    X, y = load_iris(return_X_y=True)
    assert X.shape == (150, 4)
    assert len(y) == 150
    assert sorted(np.unique(y)) == [0,1,2]

# Teste para a média e desvio padrão, além de conferir se o scaler não adicionou nem removeu colunas
def test_scaler():
    X, _ = load_iris(return_X_y=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-7)
    assert X_scaled.shape == X.shape

# Teste para assegurar que os hiperparâmetros estão de acordo com o esperado
def test_range_params():
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    assert len(C_range) == 13
    assert len(gamma_range) == 13
    assert np.all(C_range > 0)
    assert np.all(gamma_range > 0)

# Teste para assegurar que o dicionário de parâmetros possui as chaves esperadas
def test_grid_keys():
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = {"C": C_range, "gamma": gamma_range}
    assert "C" in param_grid
    assert "gamma" in param_grid

# Teste de configuração dos treinos e testes
def test_cv():
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    assert cv.n_splits == 5
    assert cv.test_size == 0.2
    assert cv.random_state == 42

# Teste de execução completa, rodando todos os blocos do código
def test_grid_run():
    X, y = load_iris(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9,3, 13)
    param_grid = {"C": C_range, "gamma": gamma_range}

    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)

    grid.fit(X_scaled, y)

    assert "C" in grid.best_params_
    assert "gamma" in grid.best_params_
    assert 0 <= grid.best_score_ <= 1