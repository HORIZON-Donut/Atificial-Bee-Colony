# Artificial Bee Colony (ABC) Visualization in 2D (Colab-ready)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# -------------------------------
# CONFIG (การตั้งค่าพารามิเตอร์)
# -------------------------------
DIM = 2                # มิติของปัญหา (2 มิติ เพราะจะทำ visualization ง่าย)
POP = 30               # จำนวนผึ้ง (จำนวนคำตอบในประชากร)
MAX_ITER = 100         # จำนวนรอบการทำงาน (iterations)
LIMIT = 20             # ถ้าแหล่งอาหารไม่ดีเกิน LIMIT รอบ จะทิ้งไป (Scout bee จะหาที่ใหม่)
LOWER, UPPER = -5.12, 5.12  # ขอบเขตค่าตัวแปร

# -------------------------------
# ฟังก์ชันเป้าหมาย (Objective Function): Rastrigin
# ใช้เป็นตัวอย่างสำหรับ optimization
# -------------------------------
def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# -------------------------------
# ฟังก์ชันช่วยสร้างประชากรเริ่มต้น (Initial population)
# -------------------------------
def init_population(pop_size, dim):
    return np.random.uniform(LOWER, UPPER, size=(pop_size, dim))

# สร้าง solution ใหม่จาก solution เดิม (ใช้ในขั้น employed/onlooker bee)
def neighbor(solution, population):
    pop_size = len(population)
    k = random.randrange(pop_size)
    while np.array_equal(population[k], solution):  # ห้ามเลือกตัวเอง
        k = random.randrange(pop_size)
    phi = np.random.uniform(-1, 1, size=solution.shape)  # ค่าแบบสุ่มระหว่าง -1 ถึง 1
    return solution + phi * (solution - population[k])   # สร้างคำตอบใหม่ใกล้ๆ ของเดิม

# -------------------------------
# ฟังก์ชันหลักในการรัน ABC
# จะบันทึกตำแหน่งผึ้งแต่ละรอบ เพื่อใช้ทำ animation
# -------------------------------
def abc_run():
    # 1) สร้างประชากรเริ่มต้น
    population = init_population(POP, DIM)
    fitness = np.array([rastrigin(ind) for ind in population])  # คำนวณค่าความเหมาะสม
    trials = np.zeros(POP, dtype=int)  # นับจำนวนครั้งที่ไม่เจอที่ดีกว่า

    # เก็บ best solution
    best_idx = np.argmin(fitness)
    best = population[best_idx].copy()
    best_f = fitness[best_idx]

    # เก็บตำแหน่งผึ้งแต่ละรอบ
    snapshots = [population.copy()]

    # 2) เริ่มวนรอบตามจำนวน iteration
    for it in range(MAX_ITER):
        # ---------------- Employed bees ----------------
        for i in range(POP):
            new_sol = neighbor(population[i], population)
            new_fit = rastrigin(new_sol)
            if new_fit < fitness[i]:
                population[i] = new_sol
                fitness[i] = new_fit
                trials[i] = 0
            else:
                trials[i] += 1

        # ---------------- Onlooker bees ----------------
        max_fit = np.max(fitness)
        probs = (max_fit - fitness) + 1e-9   # ยิ่งค่าดี ยิ่งมีโอกาสถูกเลือก
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

        # ---------------- Scout bees ----------------
        for i in range(POP):
            if trials[i] > LIMIT:
                population[i] = np.random.uniform(LOWER, UPPER, size=DIM)
                fitness[i] = rastrigin(population[i])
                trials[i] = 0

        # ---------------- Update best ----------------
        cur_best_idx = np.argmin(fitness)
        cur_best_f = fitness[cur_best_idx]
        if cur_best_f < best_f:
            best_f = cur_best_f
            best = population[cur_best_idx].copy()

        # เก็บ snapshot ของประชากรในรอบนี้
        snapshots.append(population.copy())

    return snapshots, best, best_f

# -------------------------------
# Visualization (การสร้างภาพเคลื่อนไหว)
# -------------------------------
snapshots, best, best_f = abc_run()

# สร้าง contour ของฟังก์ชัน Rastrigin
x = np.linspace(LOWER, UPPER, 200)
y = np.linspace(LOWER, UPPER, 200)
X, Y = np.meshgrid(x, y)
Z = 10*2 + (X**2 - 10*np.cos(2*np.pi*X)) + (Y**2 - 10*np.cos(2*np.pi*Y))

fig, ax = plt.subplots(figsize=(6, 6))
ax.contourf(X, Y, Z, levels=50, cmap="viridis")   # พื้นที่ฟังก์ชัน
scat = ax.scatter([], [], c="red", s=30, label="Bees")  # จุดแสดงตำแหน่งผึ้ง
ax.set_title("Artificial Bee Colony (ABC) Visualization")
ax.set_xlim(LOWER, UPPER)
ax.set_ylim(LOWER, UPPER)
ax.legend()

# ฟังก์ชันอัปเดตตำแหน่งผึ้งในแต่ละ frame ของ animation
def update(frame):
    scat.set_offsets(snapshots[frame])
    ax.set_title(f"Iteration {frame}/{MAX_ITER}")
    return scat,

# สร้าง animation
ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=200, blit=True)

# แสดง animation ใน Colab
from IPython.display import HTML
HTML(ani.to_jshtml())

