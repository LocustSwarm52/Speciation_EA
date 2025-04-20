import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# EA PARAMETERS
population_size = 50
generations     = 200
mutation_rate   = 0.9
crossover_rate  = 0.2
tournament_size = 3
elitism         = False
elitism_count   = 5

# TRAIT RANGES
x_range         = (-10, 10)
y_range         = (-10, 10)
speed_range     = (0, 1)
body_size_range = (0, 1)

# FITNESS LANDSCAPE 
def custom_landscape(x, y):
    centers = [(-6, -6), (6, -4), (-5, 5), (4, 6)]
    fitness = np.zeros_like(x)
    for cx, cy in centers:
        fitness += 10 * np.exp(-((x - cx)**2 + (y - cy)**2) / 5)
    return fitness

def calculate_calorie_use(speed):
    return speed * 0.5

def calculate_weight(body_size):
    return body_size * 2

def raw_fitness(ind):
    x, y, speed, size = ind
    landscape = custom_landscape(x, y)
    penalty   = 0.1 * calculate_calorie_use(speed) + 0.2 * calculate_weight(size)
    return landscape - penalty

# FITNESS SHARING 
def shared_fitness(ind, population, sigma=3.0, alpha=1.0):
    raw = raw_fitness(ind)
    share_sum = 0.0
    for other in population:
        d = np.hypot(ind[0]-other[0], ind[1]-other[1])
        if d < sigma:
            share_sum += 1 - (d / sigma)**alpha
    return raw / share_sum if share_sum > 0 else raw

# HELPERS 
def clamp(val, vmin, vmax):
    return max(vmin, min(vmax, val))

def clamp_ind(ind):
    x, y, s, b = ind
    return (
        clamp(x, *x_range),
        clamp(y, *y_range),
        clamp(s, *speed_range),
        clamp(b, *body_size_range)
    )

# INITIAL POPULATION 
population = [
    (random.uniform(*x_range), random.uniform(*y_range),
     random.uniform(*speed_range), random.uniform(*body_size_range))
    for _ in range(population_size)
]

# GENETIC OPERATORS 
def tournament_selection(pop):
    competitors = random.sample(pop, tournament_size)
    return max(competitors, key=lambda ind: shared_fitness(ind, pop))

def crossover(p1, p2):
    if random.random() < crossover_rate:
        return tuple((a + b) / 2 for a, b in zip(p1, p2))
    return p1

def mutate(ind):
    x, y, s, b = ind
    if random.random() < mutation_rate:
        x += random.uniform(-0.1, 0.1)
        y += random.uniform(-0.1, 0.1)
        s += random.uniform(-0.05, 0.05)
        b += random.uniform(-0.05, 0.05)
    return clamp_ind((x, y, s, b))

# PLOTTING SETUP 
fig, ax = plt.subplots(figsize=(10, 8))

# Create a grid and compute the static landscape heatmap
xs = np.linspace(*x_range, 300)
ys = np.linspace(*y_range, 300)
X, Y = np.meshgrid(xs, ys)
Z     = custom_landscape(X, Y)

# Draw the heatmap (2D “height”)
heat = ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
cbar = fig.colorbar(heat, ax=ax)
cbar.set_label("Landscape Height")

# Scatter for the evolving population
scat = ax.scatter([], [], c='r', s=10, alpha=0.8)

ax.set_title("EA with Fitness Sharing on 4‑Peak Landscape (Heatmap View)")
ax.set_xlabel("X")
ax.set_ylabel("Y")

# ANIMATION  
def update(frame):
    global population

    # sort by shared fitness for elitism
    population.sort(key=lambda ind: shared_fitness(ind, population), reverse=True)

    new_pop = []
    if elitism:
        new_pop.extend(population[:elitism_count])

    while len(new_pop) < population_size:
        p1 = tournament_selection(population)
        p2 = tournament_selection(population)
        child = crossover(p1, p2)
        child = mutate(child)
        new_pop.append(child)

    population[:] = new_pop

    # update scatter positions
    coords = np.array(population)[:, :2]
    scat.set_offsets(coords)
    return scat,

# RUN 
ani = animation.FuncAnimation(fig, update, frames=generations,
                              interval=100, blit=True)
plt.show()
