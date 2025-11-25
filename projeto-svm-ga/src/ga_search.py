import random
import math
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

class GeneticSearchSVM:
    def __init__(
        self,
        X,
        y,
        pop_size=20,
        generations=10,
        mutation_rate=0.1,
        crossover_rate=0.8,
        test_size=0.2,
        random_state=42
    ):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.random_state = random_state

        self.C_low, self.C_high = 0.1, 100
        self.gamma_low, self.gamma_high = 0.001, 1

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def random_log_param(self, low, high):
        return 10 ** random.uniform(math.log10(low), math.log10(high))

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            individual = {
                "C": self.random_log_param(self.C_low, self.C_high),
                "gamma": self.random_log_param(self.gamma_low, self.gamma_high)
            }
            population.append(individual)
        return population

    def fitness(self, individual):
        model = SVC(
            kernel='rbf',
            C=individual["C"],
            gamma=individual["gamma"]
        )
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        return accuracy_score(self.y_test, preds)

    def selection(self, population, fitness_scores):
        selected = []
        for _ in range(self.pop_size):
            i, j = random.sample(range(self.pop_size), 2)
            winner = population[i] if fitness_scores[i] > fitness_scores[j] else population[j]
            selected.append(winner)
        return selected

    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1 = {
            "C": parent1["C"],
            "gamma": parent2["gamma"]
        }
        child2 = {
            "C": parent2["C"],
            "gamma": parent1["gamma"]
        }
        return child1, child2

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            individual["C"] = self.random_log_param(self.C_low, self.C_high)
        if random.random() < self.mutation_rate:
            individual["gamma"] = self.random_log_param(self.gamma_low, self.gamma_high)
        return individual

    def run(self):
        population = self.initialize_population()
        generations_C = []
        generations_gamma = []

        for gen in range(self.generations):
            fitness_scores = [self.fitness(ind) for ind in population]
            best = population[np.argmax(fitness_scores)]
            print(f"Geração {gen + 1}/{self.generations} | 'C' atual: {best['C']} | 'Gamma' atual: {best['gamma']} | Melhor accuracy: {max(fitness_scores):.4f}")
            
            generations_C.append(best['C'])
            generations_gamma.append(best['gamma'])

            selected = self.selection(population, fitness_scores)
            
            next_population = []
            for i in range(0, self.pop_size, 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % self.pop_size]

                child1, child2 = self.crossover(parent1, parent2)

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                next_population.extend([child1, child2])

            population = next_population

        final_scores = [self.fitness(ind) for ind in population]
        best = population[np.argmax(final_scores)]
        best_acc = max(final_scores)

        print("\n=== Resultado Final GA ===")
        print("Melhor C:", best["C"])
        print("Melhor gamma:", best["gamma"])
        print("Acurácia:", best_acc)

        return best, best_acc, generations_C, generations_gamma


if __name__ == "__main__":
    from data import load_dataset

    X, y = load_dataset()

    ga = GeneticSearchSVM(X, y, pop_size=20, generations=10)

    start_time = time.time()
    best_params, best_acc, generations_C, generations_gamma  = ga.run()
    end_time = time.time()
    finish_time = end_time - start_time

    print("\nMelhores parâmetros encontrados pelo GA:")
    print(best_params)
    print("Acurácia:", best_acc)
    print(f"Tempo de execução: {end_time - start_time:.2f}s")

    generations = range(len(generations_C))

    plt.figure(figsize=(8, 4))
    plt.plot(generations, generations_C, marker='o')
    plt.title("C value over generations")
    plt.xlabel("Generation")
    plt.ylabel("C")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("1D-plot-C.png")

    plt.figure(figsize=(8, 4))
    plt.plot(generations, generations_gamma, marker='o')
    plt.title("gamma value over generations")
    plt.xlabel("Generation")
    plt.ylabel("gamma")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("1D-plot-gamma.png")

