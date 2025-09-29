# Artificial Bee Colony (ABC) Visualization in 2D (Colab-ready)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# -------------------------------
# CONFIG
# -------------------------------
DIM = 2                # ใช้ 2 มิติ เพื่อ visualize
POP = 30               # จำนวนผึ้ง
MAX_ITER = 1000         # จำนวน iteration
LIMIT = 20             # scout limit
LOWER, UPPER = -5.12, 5.12

# -------------------------------
# Objective Function: Rastrigin
# -------------------------------
def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# -------------------------------
# Helper functions
# -------------------------------
def init_population(pop_size, dim):
    return np.random.uniform(LOWER, UPPER, size=(pop_size, dim))

def neighbor(solution, population):
    pop_size = len(population)
    k = random.randrange(pop_size)
    while np.array_equal(population[k], solution):
        k = random.randrange(pop_size)
    phi = np.random.uniform(-1, 1, size=solution.shape)
    return solution + phi * (solution - population[k])

# -------------------------------
# Run ABC (but record each step)
# -------------------------------
def abc_run():
    population = init_population(POP, DIM)
    fitness = np.array([rastrigin(ind) for ind in population])
    trials = np.zeros(POP, dtype=int)

    best_idx = np.argmin(fitness)
    best = population[best_idx].copy()
    best_f = fitness[best_idx]

    snapshots = [population.copy()]  # save positions for animation

    for it in range(MAX_ITER):
        # Employed bees
        for i in range(POP):
            new_sol = neighbor(population[i], population)
            new_fit = rastrigin(new_sol)
            if new_fit < fitness[i]:
                population[i] = new_sol
                fitness[i] = new_fit
                trials[i] = 0
            else:
                trials[i] += 1

        # Onlooker bees
        max_fit = np.max(fitness)
        probs = (max_fit - fitness) + 1e-9
        probs = probs / np.sum(probs)
        onlooker_count = POP
        i, t = 0, 0
        while t < onlooker_count:
            if random.random() < probs[i]:
                new_sol = neighbor(population[i], population)
                new_fit = rastrigin(new_sol)
                if new_fit < fitness[i]:
                    population[i] = new_sol
                    fitness[i] = new_fit
                    trials[i] = 0
                else:
                    trials[i] += 1
                t += 1
            i = (i + 1) % POP

        # Scout bees
        for i in range(POP):
            if trials[i] > LIMIT:
                population[i] = np.random.uniform(LOWER, UPPER, size=DIM)
                fitness[i] = rastrigin(population[i])
                trials[i] = 0

        # Update best
        cur_best_idx = np.argmin(fitness)
        cur_best_f = fitness[cur_best_idx]
        if cur_best_f < best_f:
            best_f = cur_best_f
            best = population[cur_best_idx].copy()

        # save snapshot
        snapshots.append(population.copy())

    return snapshots, best, best_f

# -------------------------------
# Visualization
# -------------------------------
snapshots, best, best_f = abc_run()

# Prepare contour of Rastrigin
x = np.linspace(LOWER, UPPER, 200)
y = np.linspace(LOWER, UPPER, 200)
X, Y = np.meshgrid(x, y)
Z = 10*2 + (X**2 - 10*np.cos(2*np.pi*X)) + (Y**2 - 10*np.cos(2*np.pi*Y))

fig, ax = plt.subplots(figsize=(6, 6))
ax.contourf(X, Y, Z, levels=50, cmap="viridis")
scat = ax.scatter([], [], c="red", s=30, label="Bees")
ax.set_title("Artificial Bee Colony (ABC) Visualization")
ax.set_xlim(LOWER, UPPER)
ax.set_ylim(LOWER, UPPER)
ax.legend()

# Update function for animation
def update(frame):
    scat.set_offsets(snapshots[frame])
    ax.set_title(f"Iteration {frame}/{MAX_ITER}")
    return scat,

ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=200, blit=True)

# Show animation inline in Colab
from IPython.display import HTML
HTML(ani.to_jshtml())
