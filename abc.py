# Artificial Bee Colony (ABC) Visualization in 2D with Ackley Function
# Colab-Ready

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# -------------------------------
# CONFIG
# -------------------------------
DIM = 2                # ใช้ 2 มิติ เพื่อ visualize
POP = 30               # จำนวนผึ้ง
MAX_ITER = 50          # จำนวน iteration (ไม่ต้องเยอะ เดี๋ยวผึ้งไปรวมกลางหมด)
LIMIT = 20             # scout limit
LOWER, UPPER = -5, 5

# -------------------------------
# Objective Function: Ackley
SHIFT = np.array([3, 3])  # ตำแหน่ง optimum ใหม่

def shifted_ackley(x):
    x_shifted = x - SHIFT
    return -20*np.exp(-0.2*np.sqrt(0.5*(x_shifted[0]**2 + x_shifted[1]**2))) \
           - np.exp(0.5*(np.cos(2*np.pi*x_shifted[0]) + np.cos(2*np.pi*x_shifted[1]))) \
           + np.e + 20

# -------------------------------
# Helper functions
# -------------------------------
def init_population(pop_size, dim, clusters=3):
    """ สร้างประชากรแบบกระจายเป็น cluster """
    population = []
    for _ in range(clusters):
        center = np.random.uniform(LOWER, UPPER, size=dim)
        for _ in range(pop_size // clusters):
            point = center + np.random.normal(0, 0.8, size=dim)  # กระจายรอบ center
            population.append(point)
    return np.array(population)

def neighbor(solution, population):
    """ หาเพื่อนบ้าน + noise """
    pop_size = len(population)
    k = random.randrange(pop_size)
    while np.array_equal(population[k], solution):
        k = random.randrange(pop_size)
    phi = np.random.uniform(-1, 1, size=solution.shape)
    perturb = np.random.normal(0, 0.2, size=solution.shape)  # ✅ noise
    return solution + phi * (solution - population[k]) + perturb

# -------------------------------
# Run ABC (record each step)
# -------------------------------
def abc_run():
    population = init_population(POP, DIM)
    # Corrected: Use shifted_ackley instead of ackley
    fitness = np.array([shifted_ackley(ind) for ind in population])
    trials = np.zeros(POP, dtype=int)

    best_idx = np.argmin(fitness)
    best = population[best_idx].copy()
    best_f = fitness[best_idx]

    snapshots = [population.copy()]  # save positions for animation

    for it in range(MAX_ITER):
        # Employed bees
        for i in range(POP):
            new_sol = neighbor(population[i], population)
            # Corrected: Use shifted_ackley instead of ackley
            new_fit = shifted_ackley(new_sol)
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
                # Corrected: Use shifted_ackley instead of ackley
                new_fit = shifted_ackley(new_sol)
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
                # Corrected: Use shifted_ackley instead of ackley
                fitness[i] = shifted_ackley(population[i])
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

# Prepare contour of Ackley
x = np.linspace(LOWER, UPPER, 200)
y = np.linspace(LOWER, UPPER, 200)
X, Y = np.meshgrid(x, y)
# Corrected: Apply shifted_ackley element-wise to the meshgrid
Z = np.array([[shifted_ackley(np.array([xi, yi])) for xi in x] for yi in y])


fig, ax = plt.subplots(figsize=(6, 6))
ax.contourf(X, Y, Z, levels=80, cmap="plasma")

scat = ax.scatter([], [], c="red", s=30, label="Bees")
ax.set_title("Artificial Bee Colony (ABC) on Ackley Function")
ax.set_xlim(LOWER, UPPER)
ax.set_ylim(LOWER, UPPER)
ax.set_aspect("equal")  # ✅ ทำให้ field ดูไม่บี้
ax.legend()

# Update function for animation
def update(frame):
    scat.set_offsets(snapshots[frame])
    ax.set_title(f"Iteration {frame}/{MAX_ITER}")
    return scat,

ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=300, blit=True)

# Show animation inline in Colab
from IPython.display import HTML
HTML(ani.to_jshtml())