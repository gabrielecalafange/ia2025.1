import pytest
import numpy as np
import random
from sklearn.datasets import make_classification
import importlib.util
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
ga_search_path = os.path.join(current_dir, "ga-search.py")

spec = importlib.util.spec_from_file_location("ga_search", ga_search_path)
ga_search_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ga_search_module)
GeneticSearchSVM = ga_search_module.GeneticSearchSVM


class TestGeneticSearchSVM:
    
    @pytest.fixture
    def sample_data(self):
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_classes=2,
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def ga_instance(self, sample_data):
        X, y = sample_data
        return GeneticSearchSVM(
            X, y, 
            pop_size=10, 
            generations=5,
            random_state=42
        )
    
    def test_ga_init(self, sample_data):
        X, y = sample_data
        ga = GeneticSearchSVM(X, y, pop_size=15, generations=8)
        
        assert ga.pop_size == 15
        assert ga.generations == 8
        assert ga.c_range == (0.1, 1000)
        assert ga.gamma_range == (0.0001, 1)
        assert ga.mutation_rate == 0.1
        assert ga.crossover_rate == 0.8
        assert ga.random_state == 42
        
        assert len(ga.X_train) + len(ga.X_test) == len(X)
        assert len(ga.y_train) + len(ga.y_test) == len(y)
    
    def test_random_param(self, ga_instance):
        random.seed(42)
        
        params = [ga_instance.random_param((0.1, 1.0)) for _ in range(10)]
        
        for param in params:
            assert 0.1 <= param <= 1.0
        
        params_2 = [ga_instance.random_param((10, 100)) for _ in range(5)]
        for param in params_2:
            assert 10 <= param <= 100
    
    def test_initialize_population(self, ga_instance):
        population = ga_instance.initialize_population()
        
        assert len(population) == ga_instance.pop_size
        
        for individual in population:
            assert "C" in individual
            assert "gamma" in individual
            assert ga_instance.c_range[0] <= individual["C"] <= ga_instance.c_range[1]
            assert ga_instance.gamma_range[0] <= individual["gamma"] <= ga_instance.gamma_range[1]
        
    def test_population_diversity(self, ga_instance):
        population = ga_instance.initialize_population()

        c_values = [ind["C"] for ind in population]
        gamma_values = [ind["gamma"] for ind in population]
        
        assert len(set(c_values)) > 1 or len(set(gamma_values)) > 1
    
    def test_fitness(self, ga_instance):
        individual = {"C": 1.0, "gamma": 0.01}
        
        fitness_score = ga_instance.fitness(individual)
        
        assert 0 <= fitness_score <= 1
        assert isinstance(fitness_score, float)
    
    def test_selection(self, ga_instance):
        population = [
            {"C": 1.0, "gamma": 0.01},
            {"C": 2.0, "gamma": 0.02},
            {"C": 3.0, "gamma": 0.03}
        ]
        fitness_scores = [0.8, 0.9, 0.7]
        
        original_pop_size = ga_instance.pop_size
        ga_instance.pop_size = 3
        
        selected = ga_instance.selection(population, fitness_scores)
        
        ga_instance.pop_size = original_pop_size
        
        assert len(selected) == 3
        
        for individual in selected:
            assert individual in population
    
    def test_crossover_no_crossover(self, ga_instance):
        parent1 = {"C": 1.0, "gamma": 0.01}
        parent2 = {"C": 2.0, "gamma": 0.02}
        
        original_random = random.random
        random.random = lambda: 0.9 
        
        try:
            child1, child2 = ga_instance.crossover(parent1, parent2)
            
            assert child1 == parent1
            assert child2 == parent2
            assert child1 is not parent1  
            assert child2 is not parent2
        finally:
            random.random = original_random
    
    def test_crossover_with_crossover(self, ga_instance):
        parent1 = {"C": 1.0, "gamma": 0.01}
        parent2 = {"C": 2.0, "gamma": 0.02}
        
        original_random = random.random
        random.random = lambda: 0.5 
        
        try:
            child1, child2 = ga_instance.crossover(parent1, parent2)
            
            assert child1["C"] == parent1["C"]
            assert child1["gamma"] == parent2["gamma"]
            assert child2["C"] == parent2["C"]
            assert child2["gamma"] == parent1["gamma"]
        finally:
            random.random = original_random
    
    def test_mutate_no_mutation(self, ga_instance):
        individual = {"C": 1.0, "gamma": 0.01}
        original = individual.copy()
        
        original_random = random.random
        random.random = lambda: 0.9 
        
        try:
            mutated = ga_instance.mutate(individual)
            assert mutated == original
        finally:
            random.random = original_random
    
    def test_mutate_with_mutation(self, ga_instance):
        individual = {"C": 1.0, "gamma": 0.01}
        original = individual.copy()
        
        original_random = random.random
        random.random = lambda: 0.05  
        
        try:
            mutated = ga_instance.mutate(individual)
            assert (mutated["C"] != original["C"] or 
                   mutated["gamma"] != original["gamma"])
        finally:
            random.random = original_random
    
    def test_run_with_small_parameters(self, sample_data):
        X, y = sample_data
        
        ga = GeneticSearchSVM(
            X, y, 
            pop_size=4, 
            generations=2,
            random_state=42
        )
        
        best_params, best_acc = ga.run()
        
        assert isinstance(best_params, dict)
        assert "C" in best_params
        assert "gamma" in best_params
        assert ga.c_range[0] <= best_params["C"] <= ga.c_range[1]
        assert ga.gamma_range[0] <= best_params["gamma"] <= ga.gamma_range[1]
        assert 0 <= best_acc <= 1
    
    def test_custom_parameters(self, sample_data):
        X, y = sample_data
        
        ga = GeneticSearchSVM(
            X, y,
            pop_size=8,
            generations=3,
            c_range=(1, 100),
            gamma_range=(0.01, 0.1),
            mutation_rate=0.2,
            crossover_rate=0.7,
            test_size=0.3,
            random_state=123
        )
        
        assert ga.pop_size == 8
        assert ga.generations == 3
        assert ga.c_range == (1, 100)
        assert ga.gamma_range == (0.01, 0.1)
        assert ga.mutation_rate == 0.2
        assert ga.crossover_rate == 0.7
        assert ga.random_state == 123
    
    def test_parameter_ranges(self, ga_instance):
        for _ in range(20):
            c_param = ga_instance.random_param(ga_instance.c_range)
            gamma_param = ga_instance.random_param(ga_instance.gamma_range)
            
            assert ga_instance.c_range[0] <= c_param <= ga_instance.c_range[1]
            assert ga_instance.gamma_range[0] <= gamma_param <= ga_instance.gamma_range[1]
    
    def test_population_evolution(self, sample_data):
        X, y = sample_data
        
        ga = GeneticSearchSVM(X, y, pop_size=6, generations=3, random_state=42)
        ga.initialize_population()
        
        best_params, best_acc = ga.run()
        
        assert isinstance(best_params, dict)
        assert isinstance(best_acc, float)
        assert 0 <= best_acc <= 1

    def test_fitness_deterministic(self, ga_instance):
        individual = {"C": 1.0, "gamma": 0.01}
        
        fitness1 = ga_instance.fitness(individual)
        fitness2 = ga_instance.fitness(individual)
        
        assert fitness1 == fitness2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])