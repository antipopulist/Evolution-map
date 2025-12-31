import copy
import time
from dataclasses import dataclass

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


@dataclass
class TerrainDef:
    name: str
    color: str
    max_resource: float
    regen: float
    passable: bool


TERRAIN = {
    0: TerrainDef("plains", "#c2b280", 3.0, 0.12, True),
    1: TerrainDef("forest", "#3c7d3e", 5.0, 0.20, True),
    2: TerrainDef("water", "#4a76a8", 0.0, 0.0, False),
    3: TerrainDef("mountain", "#7a7a7a", 0.0, 0.0, False),
}

MIN_BASE_METABOLISM = 0.35
MAX_BASE_METABOLISM = 1.2
DEFAULT_SPEED_COST = 0.07
DEFAULT_VISION_COST = 0.03
DEFAULT_EAT_RATE_COST = 0.12
EAT_RATE_MIN = 0.2
EAT_RATE_MAX = 3.0
EAT_RATE_BASELINE = 1.0
EAT_RATE_MUTATION_STD = 0.15


def hex_to_rgb(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip("#")
    return np.array(
        [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)], dtype=float
    ) / 255.0


def smooth_field(size: int, rng: np.random.Generator, passes: int) -> np.ndarray:
    field = rng.random((size, size))
    for _ in range(max(0, int(passes))):
        field = (
            field
            + np.roll(field, 1, 0)
            + np.roll(field, -1, 0)
            + np.roll(field, 1, 1)
            + np.roll(field, -1, 1)
            + np.roll(np.roll(field, 1, 0), 1, 1)
            + np.roll(np.roll(field, 1, 0), -1, 1)
            + np.roll(np.roll(field, -1, 0), 1, 1)
            + np.roll(np.roll(field, -1, 0), -1, 1)
        ) / 9.0
    return field


def normalize_weights(weights: dict) -> dict:
    total = float(sum(weights.values()))
    if total <= 0:
        equal = 1.0 / max(1, len(weights))
        return {key: equal for key in weights}
    return {key: value / total for key, value in weights.items()}


def generate_clustered_terrain(
    size: int, rng: np.random.Generator, terrain_weights: dict, smooth_steps: int
) -> np.ndarray:
    weights = normalize_weights(terrain_weights)
    water_frac = weights.get(2, 0.0)
    mountain_frac = weights.get(3, 0.0)
    forest_frac = weights.get(1, 0.0)

    height = smooth_field(size, rng, smooth_steps)
    moisture = smooth_field(size, rng, smooth_steps)

    terrain = np.zeros((size, size), dtype=int)

    water_mask = np.zeros((size, size), dtype=bool)
    if water_frac > 0:
        water_thresh = np.quantile(height, water_frac)
        water_mask = height <= water_thresh
        terrain[water_mask] = 2

    mountain_mask = np.zeros((size, size), dtype=bool)
    if mountain_frac > 0:
        mountain_thresh = np.quantile(height, 1 - mountain_frac)
        mountain_mask = (height >= mountain_thresh) & ~water_mask
        terrain[mountain_mask] = 3

    land_mask = ~(water_mask | mountain_mask)
    if forest_frac > 0 and land_mask.any():
        land_frac = float(land_mask.mean())
        desired_forest = min(forest_frac, land_frac)
        if desired_forest > 0:
            forest_in_land = desired_forest / land_frac
            if forest_in_land >= 1:
                forest_mask = land_mask
            else:
                moisture_land = moisture[land_mask]
                forest_thresh = np.quantile(moisture_land, 1 - forest_in_land)
                forest_mask = land_mask & (moisture >= forest_thresh)
            terrain[forest_mask] = 1

    return terrain


def init_map(
    size: int,
    rng: np.random.Generator,
    terrain_weights: dict,
    smooth_steps: int,
) -> dict:
    terrain = generate_clustered_terrain(size, rng, terrain_weights, smooth_steps)

    max_resource = np.zeros((size, size), dtype=float)
    regen = np.zeros((size, size), dtype=float)
    passable = np.zeros((size, size), dtype=bool)
    for t_id, t_def in TERRAIN.items():
        mask = terrain == t_id
        max_resource[mask] = t_def.max_resource
        regen[mask] = t_def.regen
        passable[mask] = t_def.passable

    resource = rng.random((size, size)) * max_resource
    return {
        "terrain": terrain,
        "resource": resource,
        "max_resource": max_resource,
        "regen": regen,
        "passable": passable,
    }


def init_agents(
    count: int,
    energy: float,
    base_eat_rate: float,
    rng: np.random.Generator,
    passable: np.ndarray,
    costs: dict,
) -> list:
    positions = np.argwhere(passable)
    if len(positions) == 0:
        return []
    indices = rng.choice(len(positions), size=count, replace=True)
    agents = []
    for i, idx in enumerate(indices):
        x, y = positions[idx]
        agents.append(
            {
                "id": i,
                "x": int(x),
                "y": int(y),
                "energy": float(energy),
                "speed": int(rng.integers(1, 3)),
                "vision": int(rng.integers(2, 6)),
                "eat_rate": float(base_eat_rate),
                "base_metabolism": float(rng.uniform(MIN_BASE_METABOLISM, 0.8)),
            }
        )
        agent = agents[-1]
        agent["metabolism"] = compute_metabolism(
            agent["speed"], agent["vision"], agent["eat_rate"], agent["base_metabolism"], costs
        )
    return agents


def init_predators(
    count: int, energy: float, rng: np.random.Generator, passable: np.ndarray, costs: dict
) -> list:
    positions = np.argwhere(passable)
    if len(positions) == 0 or count <= 0:
        return []
    indices = rng.choice(len(positions), size=count, replace=True)
    predators = []
    for i, idx in enumerate(indices):
        x, y = positions[idx]
        predators.append(
            {
                "id": i,
                "x": int(x),
                "y": int(y),
                "energy": float(energy),
                "speed": int(rng.integers(1, 4)),
                "vision": 4,
                "base_metabolism": float(rng.uniform(MIN_BASE_METABOLISM, 0.9)),
            }
        )
        predator = predators[-1]
        predator["metabolism"] = compute_metabolism(
            predator["speed"],
            predator["vision"],
            EAT_RATE_BASELINE,
            predator["base_metabolism"],
            costs,
        )
    return predators


def metabolism_penalty(speed: int, vision: int, eat_rate: float, costs: dict) -> float:
    speed_penalty = max(0, int(speed) - 1) * costs["speed"]
    vision_penalty = max(0, int(vision) - 2) * costs["vision"]
    eat_penalty = max(0.0, float(eat_rate) - EAT_RATE_BASELINE) * costs["eat_rate"]
    return float(speed_penalty + vision_penalty + eat_penalty)


def compute_metabolism(
    speed: int, vision: int, eat_rate: float, base_metabolism: float, costs: dict
) -> float:
    penalty = metabolism_penalty(speed, vision, eat_rate, costs)
    return float(base_metabolism + penalty)


def ensure_metabolism(agent: dict, costs: dict) -> None:
    eat_rate = float(agent.get("eat_rate", EAT_RATE_BASELINE))
    if "base_metabolism" not in agent:
        penalty = metabolism_penalty(agent["speed"], agent["vision"], eat_rate, costs)
        base = float(max(MIN_BASE_METABOLISM, agent["metabolism"] - penalty))
        agent["base_metabolism"] = min(base, MAX_BASE_METABOLISM)
    agent["metabolism"] = compute_metabolism(
        agent["speed"], agent["vision"], eat_rate, agent["base_metabolism"], costs
    )


def pick_target(
    x: int,
    y: int,
    vision: int,
    resource: np.ndarray,
    passable: np.ndarray,
    rng: np.random.Generator,
) -> tuple:
    size = resource.shape[0]
    r = max(1, int(vision))
    x0 = max(0, x - r)
    x1 = min(size - 1, x + r)
    y0 = max(0, y - r)
    y1 = min(size - 1, y + r)

    res_sub = resource[x0 : x1 + 1, y0 : y1 + 1]
    pass_sub = passable[x0 : x1 + 1, y0 : y1 + 1]
    if not pass_sub.any():
        return x, y

    masked = np.where(pass_sub, res_sub, -1.0)
    best = masked.max()
    if best <= 0:
        return x, y
    candidates = np.argwhere(masked == best)
    cx, cy = candidates[rng.integers(0, len(candidates))]
    return x0 + int(cx), y0 + int(cy)


def random_neighbor(
    x: int, y: int, passable: np.ndarray, rng: np.random.Generator
) -> tuple:
    size = passable.shape[0]
    candidates = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size and passable[nx, ny]:
            candidates.append((nx, ny))
    if not candidates:
        return x, y
    return candidates[rng.integers(0, len(candidates))]


def find_prey_target(
    x: int, y: int, vision: int, prey: list, rng: np.random.Generator
) -> tuple | None:
    best_dist = None
    candidates = []
    for agent in prey:
        dist = abs(agent["x"] - x) + abs(agent["y"] - y)
        if dist <= vision:
            if best_dist is None or dist < best_dist:
                best_dist = dist
                candidates = [(agent["x"], agent["y"])]
            elif dist == best_dist:
                candidates.append((agent["x"], agent["y"]))
    if not candidates:
        return None
    return candidates[rng.integers(0, len(candidates))]


def move_toward(
    x: int,
    y: int,
    target: tuple,
    speed: int,
    passable: np.ndarray,
    rng: np.random.Generator,
) -> tuple:
    size = passable.shape[0]
    steps = max(1, int(speed))
    tx, ty = target
    cx, cy = x, y
    for _ in range(steps):
        dx = tx - cx
        dy = ty - cy
        if dx == 0 and dy == 0:
            break
        if abs(dx) >= abs(dy):
            step_x, step_y = int(np.sign(dx)), 0
        else:
            step_x, step_y = 0, int(np.sign(dy))
        nx, ny = cx + step_x, cy + step_y
        if 0 <= nx < size and 0 <= ny < size and passable[nx, ny]:
            cx, cy = nx, ny
        else:
            cx, cy = random_neighbor(cx, cy, passable, rng)
            break
    return cx, cy


def mutate(agent: dict, rng: np.random.Generator, rate: float, costs: dict) -> dict:
    child = agent.copy()
    base = float(child.get("base_metabolism", child["metabolism"]))
    eat_rate = float(child.get("eat_rate", EAT_RATE_BASELINE))
    if rng.random() < rate:
        child["speed"] = int(np.clip(child["speed"] + rng.choice([-1, 1]), 1, 4))
    if rng.random() < rate:
        child["vision"] = int(np.clip(child["vision"] + rng.choice([-1, 1]), 1, 8))
    if "eat_rate" in child and rng.random() < rate:
        eat_rate = float(
            np.clip(eat_rate + rng.normal(0, EAT_RATE_MUTATION_STD), EAT_RATE_MIN, EAT_RATE_MAX)
        )
        child["eat_rate"] = eat_rate
    if rng.random() < rate:
        base += rng.normal(0, 0.08)
    base = float(np.clip(base, MIN_BASE_METABOLISM, MAX_BASE_METABOLISM))
    child["base_metabolism"] = base
    child["metabolism"] = compute_metabolism(
        child["speed"], child["vision"], eat_rate, base, costs
    )
    return child


def step_sim(state: dict, params: dict) -> dict:
    world = state["world"]
    agents = state["agents"]
    predators = state["predators"]
    rng = state["rng"]
    costs = params["metabolism_costs"]
    passable_count = int(world["passable"].sum())
    capacity = max(1, int(params["capacity_ratio"] * passable_count))
    density = len(agents) / capacity
    repro_chance = params["prey_repro_scale"] * min(1.0, max(0.0, 1.6 - density))

    world["resource"] = np.minimum(
        world["resource"] + world["regen"] * params["regen_scale"],
        world["max_resource"],
    )

    new_agents = []
    offspring = []
    next_id = state["next_id"]

    for agent in agents:
        if "eat_rate" not in agent:
            agent["eat_rate"] = params["base_eat_rate"]
        ensure_metabolism(agent, costs)
        target = pick_target(
            agent["x"],
            agent["y"],
            agent["vision"],
            world["resource"],
            world["passable"],
            rng,
        )
        if target == (agent["x"], agent["y"]):
            nx, ny = random_neighbor(agent["x"], agent["y"], world["passable"], rng)
        else:
            nx, ny = move_toward(
                agent["x"],
                agent["y"],
                target,
                agent["speed"],
                world["passable"],
                rng,
            )

        agent["x"], agent["y"] = nx, ny

        available = world["resource"][nx, ny]
        eaten = min(available, agent["eat_rate"])
        world["resource"][nx, ny] -= eaten
        agent["energy"] += eaten * params["energy_gain"]
        agent["energy"] -= agent["metabolism"]
        ate_food = eaten > 0.0

        if (
            agent["energy"] >= params["repro_threshold"]
            and ate_food
            and rng.random() < repro_chance
        ):
            agent["energy"] *= 0.5
            child = mutate(agent, rng, params["mutation_rate"], costs)
            child["id"] = next_id
            next_id += 1
            child["energy"] = agent["energy"]
            offspring.append(child)

        if agent["energy"] > 0:
            new_agents.append(agent)

    prey = new_agents + offspring
    prey_count = len(prey)
    predator_capacity = max(1, int(params["predator_capacity_ratio"] * passable_count))
    predator_density = len(predators) / predator_capacity
    pred_repro_base = min(1.0, max(0.0, 2.0 - predator_density))
    if len(predators) == 0:
        prey_factor = 1.0
    else:
        prey_factor = min(
            1.0,
            prey_count / max(1.0, params["prey_per_predator"] * len(predators)),
        )
    pred_repro_chance = pred_repro_base * prey_factor
    new_predators = []
    pred_offspring = []
    next_pred_id = state["next_pred_id"]

    for predator in predators:
        ensure_metabolism(predator, costs)
        ate_prey = False
        target = find_prey_target(
            predator["x"],
            predator["y"],
            predator["vision"],
            prey,
            rng,
        )
        if target is None:
            nx, ny = random_neighbor(
                predator["x"], predator["y"], world["passable"], rng
            )
        else:
            nx, ny = move_toward(
                predator["x"],
                predator["y"],
                target,
                predator["speed"],
                world["passable"],
                rng,
            )

        predator["x"], predator["y"] = nx, ny

        for idx, agent in enumerate(prey):
            if agent["x"] == nx and agent["y"] == ny:
                if rng.random() < params["predator_kill_chance"]:
                    predator["energy"] += params["predator_energy_gain"]
                    ate_prey = True
                    prey.pop(idx)
                else:
                    ex, ey = random_neighbor(
                        agent["x"], agent["y"], world["passable"], rng
                    )
                    agent["x"], agent["y"] = ex, ey
                break

        predator["energy"] -= predator["metabolism"]

        if (
            predator["energy"] >= params["predator_repro_threshold"]
            and ate_prey
            and rng.random() < pred_repro_chance
        ):
            predator["energy"] *= 0.5
            child = mutate(predator, rng, params["mutation_rate"], costs)
            child["id"] = next_pred_id
            next_pred_id += 1
            child["energy"] = predator["energy"]
            pred_offspring.append(child)

        if predator["energy"] > 0:
            new_predators.append(predator)

    state["agents"] = prey
    state["predators"] = new_predators + pred_offspring
    state["next_pred_id"] = next_pred_id
    state["next_id"] = next_id
    state["step"] += 1
    return state


def render_map(world: dict, agents: list, predators: list) -> plt.Figure:
    size = world["terrain"].shape[0]
    image = np.zeros((size, size, 3), dtype=float)
    for t_id, t_def in TERRAIN.items():
        mask = world["terrain"] == t_id
        base = hex_to_rgb(t_def.color)
        if t_def.passable and t_def.max_resource > 0:
            intensity = world["resource"] / np.maximum(world["max_resource"], 1e-6)
            shade = 0.65 + 0.35 * intensity
            image[mask] = base * shade[mask][..., None]
        else:
            image[mask] = base

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, origin="lower")
    if agents:
        xs = [a["y"] for a in agents]
        ys = [a["x"] for a in agents]
        ax.scatter(xs, ys, s=18, c="#c23b22", edgecolors="black", linewidths=0.3)
    if predators:
        xs = [p["y"] for p in predators]
        ys = [p["x"] for p in predators]
        ax.scatter(xs, ys, s=36, c="#0b3c5d", marker="^", edgecolors="white", linewidths=0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Evolution Map")
    plt.tight_layout()
    return fig


def summarize_population(agents: list) -> dict:
    if not agents:
        return {"pop": 0, "speed": 0, "vision": 0, "metabolism": 0}
    speeds = [a["speed"] for a in agents]
    visions = [a["vision"] for a in agents]
    metab = [a["metabolism"] for a in agents]
    return {
        "pop": len(agents),
        "speed": float(np.mean(speeds)),
        "vision": float(np.mean(visions)),
        "metabolism": float(np.mean(metab)),
    }


def summarize_traits(agents: list, costs: dict) -> dict:
    if not agents:
        return {
            "speed": 0.0,
            "vision": 0.0,
            "eat_rate": 0.0,
            "speed_cost": 0.0,
            "vision_cost": 0.0,
            "eat_rate_cost": 0.0,
        }
    speeds = [a["speed"] for a in agents]
    visions = [a["vision"] for a in agents]
    eat_rates = [a.get("eat_rate", EAT_RATE_BASELINE) for a in agents]
    speed_costs = [max(0, s - 1) * costs["speed"] for s in speeds]
    vision_costs = [max(0, v - 2) * costs["vision"] for v in visions]
    eat_costs = [max(0.0, e - EAT_RATE_BASELINE) * costs["eat_rate"] for e in eat_rates]
    return {
        "speed": float(np.mean(speeds)),
        "vision": float(np.mean(visions)),
        "eat_rate": float(np.mean(eat_rates)),
        "speed_cost": float(np.mean(speed_costs)),
        "vision_cost": float(np.mean(vision_costs)),
        "eat_rate_cost": float(np.mean(eat_costs)),
    }


def build_trait_series(history: list, series_keys: dict) -> list:
    data = []
    for row in history:
        step = row.get("step", 0)
        for label, key in series_keys.items():
            data.append(
                {"step": step, "series": label, "value": float(row.get(key, 0.0))}
            )
    return data


def build_trait_chart(history: list, series_keys: dict) -> alt.Chart:
    data = build_trait_series(history, series_keys)
    if not data:
        return alt.Chart({"step": [], "series": [], "value": []}).mark_line()
    domain = list(series_keys.keys())
    color_map = {
        "Prey speed": "#b45f06",
        "Prey vision": "#e69138",
        "Prey eat": "#f6b26b",
        "Pred speed": "#0b3c5d",
        "Pred vision": "#1e5aa8",
        "Pred eat": "#8db3e2",
    }
    color_range = [color_map[label] for label in domain]
    base = alt.Chart(alt.Data(values=data))
    color = alt.Color(
        "series:N",
        scale=alt.Scale(
            domain=domain,
            range=color_range,
        ),
        legend=alt.Legend(columns=3, orient="bottom"),
    )

    nearest = alt.selection_point(
        nearest=True, on="mouseover", fields=["step"], empty="none"
    )

    line = base.mark_line().encode(
        x=alt.X("step:Q", title="Step"),
        y=alt.Y("value:Q", title="Avg trait value"),
        color=color,
    )

    selectors = base.mark_point(opacity=0, size=220).encode(
        x="step:Q",
        y="value:Q",
    ).add_params(nearest)

    points = base.mark_point(size=70).encode(
        x="step:Q",
        y="value:Q",
        color=color,
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip("series:N", title="Series"),
            alt.Tooltip("value:Q", title="Value", format=".3f"),
            alt.Tooltip("step:Q", title="Step"),
        ],
    )

    return alt.layer(line, selectors, points)


def ensure_state(params: dict, regen_params: dict) -> None:
    if "state" in st.session_state:
        return
    rng = np.random.default_rng(params["seed"])
    world = init_map(
        params["size"],
        rng,
        params["terrain_weights"],
        params["smooth_steps"],
    )
    agents = init_agents(
        params["initial_pop"],
        params["initial_energy"],
        params["base_eat_rate"],
        rng,
        world["passable"],
        params["metabolism_costs"],
    )
    predators = init_predators(
        params["initial_predators"],
        params["predator_initial_energy"],
        rng,
        world["passable"],
        params["metabolism_costs"],
    )
    st.session_state.state = {
        "world": world,
        "agents": agents,
        "predators": predators,
        "step": 0,
        "next_id": len(agents),
        "next_pred_id": len(predators),
        "rng": rng,
        "history": [],
    }
    st.session_state.applied_regen_params = copy.deepcopy(regen_params)


def rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def main() -> None:
    st.set_page_config(page_title="Evolution Map", layout="wide")
    st.title("Evolution Map")

    if "auto_step" not in st.session_state:
        st.session_state.auto_step = False

    with st.sidebar:
        sidebar_disabled = st.session_state.auto_step
        st.header("World")
        size = st.slider(
            "Map size",
            20,
            80,
            50,
            5,
            help="Grid size in tiles (NxN). Larger maps are slower.",
            disabled=sidebar_disabled,
        )
        seed = st.number_input(
            "Seed",
            value=42,
            step=1,
            help="Random seed for map and initial traits.",
            disabled=sidebar_disabled,
        )
        initial_pop = st.slider(
            "Initial population",
            10,
            200,
            80,
            10,
            help="Starting number of prey.",
            disabled=sidebar_disabled,
        )
        initial_energy = st.slider(
            "Initial energy",
            2.0,
            10.0,
            5.0,
            0.5,
            help="Starting energy for each prey.",
            disabled=sidebar_disabled,
        )
        regen_scale = st.slider(
            "Resource regen scale",
            0.1,
            2.0,
            0.3,
            0.1,
            help="Multiplier for terrain resource regrowth.",
            disabled=sidebar_disabled,
        )

        st.header("Evolution")
        repro_threshold = st.slider(
            "Reproduction threshold",
            4.0,
            12.0,
            8.0,
            0.5,
            help="Energy needed to reproduce.",
            disabled=sidebar_disabled,
        )
        mutation_rate = st.slider(
            "Mutation rate",
            0.0,
            1.0,
            0.5,
            0.01,
            help="Chance of each trait mutating.",
            disabled=sidebar_disabled,
        )
        base_eat_rate = st.slider(
            "Base eat per step",
            EAT_RATE_MIN,
            EAT_RATE_MAX,
            1.0,
            0.1,
            help="Starting max food per step for prey (mutates).",
            disabled=sidebar_disabled,
        )
        energy_gain = st.slider(
            "Energy gain per food",
            0.5,
            3.0,
            1.0,
            0.1,
            help="Energy gained per unit of food.",
            disabled=sidebar_disabled,
        )
        steps_per_tick = st.slider(
            "Steps per tick",
            1,
            20,
            1,
            1,
            help="Simulation steps per click/auto tick.",
            disabled=sidebar_disabled,
        )
        capacity_ratio = st.slider(
            "Carrying capacity (ratio of passable tiles)",
            0.2,
            1.5,
            0.8,
            0.05,
            help="Soft cap for prey reproduction.",
            disabled=sidebar_disabled,
        )
        prey_repro_scale = st.slider(
            "Prey reproduction chance scale",
            0.1,
            1.0,
            0.6,
            0.05,
            help="Scales prey reproduction probability.",
            disabled=sidebar_disabled,
        )

        st.header("Metabolism costs")
        speed_cost = st.slider(
            "Speed cost",
            0.0,
            0.2,
            DEFAULT_SPEED_COST,
            0.01,
            help="Metabolism cost per speed above 1.",
            disabled=sidebar_disabled,
        )
        vision_cost = st.slider(
            "Vision cost",
            0.0,
            0.1,
            DEFAULT_VISION_COST,
            0.01,
            help="Metabolism cost per vision above 2.",
            disabled=sidebar_disabled,
        )
        eat_rate_cost = st.slider(
            "Eat rate cost",
            0.0,
            0.4,
            DEFAULT_EAT_RATE_COST,
            0.01,
            help="Metabolism cost per eat rate above 1.0.",
            disabled=sidebar_disabled,
        )

        st.header("Predators")
        initial_predators = st.slider(
            "Initial predators",
            0,
            60,
            8,
            1,
            help="Starting number of predators.",
            disabled=sidebar_disabled,
        )
        predator_initial_energy = st.slider(
            "Predator initial energy",
            2.0,
            12.0,
            14.0,
            0.5,
            help="Starting energy for each predator.",
            disabled=sidebar_disabled,
        )
        predator_repro_threshold = st.slider(
            "Predator reproduction threshold",
            4.0,
            16.0,
            15.0,
            0.5,
            help="Energy needed for predator reproduction.",
            disabled=sidebar_disabled,
        )
        predator_energy_gain = st.slider(
            "Energy gain per prey",
            0.5,
            4.0,
            2.0,
            0.1,
            help="Energy gained on a kill.",
            disabled=sidebar_disabled,
        )
        predator_kill_chance = st.slider(
            "Predator kill chance",
            0.1,
            1.0,
            0.65,
            0.05,
            help="Chance to kill prey on contact.",
            disabled=sidebar_disabled,
        )
        predator_capacity_ratio = st.slider(
            "Predator carrying capacity (ratio of passable tiles)",
            0.05,
            0.6,
            0.2,
            0.01,
            help="Soft cap for predator reproduction.",
            disabled=sidebar_disabled,
        )
        prey_per_predator = st.slider(
            "Prey per predator for growth",
            1.0,
            10.0,
            4.0,
            0.5,
            help="Prey availability needed for predator reproduction.",
            disabled=sidebar_disabled,
        )

        st.header("Terrain mix")
        plains = st.slider(
            "Plains",
            0.1,
            0.9,
            0.5,
            0.05,
            help="Relative share of plains terrain.",
            disabled=sidebar_disabled,
        )
        forest = st.slider(
            "Forest",
            0.05,
            0.7,
            0.25,
            0.05,
            help="Relative share of forest terrain.",
            disabled=sidebar_disabled,
        )
        water = st.slider(
            "Water",
            0.0,
            0.5,
            0.15,
            0.05,
            help="Relative share of water terrain.",
            disabled=sidebar_disabled,
        )
        mountain = st.slider(
            "Mountain",
            0.0,
            0.5,
            0.0,
            0.05,
            help="Relative share of mountain terrain.",
            disabled=sidebar_disabled,
        )
        smooth_steps = st.slider(
            "Terrain clustering",
            0,
            20,
            10,
            1,
            help="Higher values create larger clusters.",
            disabled=sidebar_disabled,
        )

    terrain_weights = {0: plains, 1: forest, 2: water, 3: mountain}

    regen_params = {
        "size": size,
        "seed": seed,
        "initial_pop": initial_pop,
        "initial_energy": initial_energy,
        "initial_predators": initial_predators,
        "predator_initial_energy": predator_initial_energy,
        "terrain_weights": terrain_weights,
        "smooth_steps": smooth_steps,
        "base_eat_rate": base_eat_rate,
    }

    params = {
        "size": size,
        "seed": seed,
        "initial_pop": initial_pop,
        "initial_energy": initial_energy,
        "regen_scale": regen_scale,
        "repro_threshold": repro_threshold,
        "mutation_rate": mutation_rate,
        "base_eat_rate": base_eat_rate,
        "energy_gain": energy_gain,
        "steps_per_tick": steps_per_tick,
        "capacity_ratio": capacity_ratio,
        "prey_repro_scale": prey_repro_scale,
        "metabolism_costs": {
            "speed": speed_cost,
            "vision": vision_cost,
            "eat_rate": eat_rate_cost,
        },
        "initial_predators": initial_predators,
        "predator_initial_energy": predator_initial_energy,
        "predator_repro_threshold": predator_repro_threshold,
        "predator_energy_gain": predator_energy_gain,
        "predator_kill_chance": predator_kill_chance,
        "predator_capacity_ratio": predator_capacity_ratio,
        "prey_per_predator": prey_per_predator,
        "terrain_weights": terrain_weights,
        "smooth_steps": smooth_steps,
    }

    ensure_state(params, regen_params)
    if "applied_regen_params" not in st.session_state:
        st.session_state.applied_regen_params = copy.deepcopy(regen_params)

    col1, col2 = st.columns([2, 1])
    with col1:
        state = st.session_state.state
        fig = render_map(state["world"], state["agents"], state["predators"])
        st.pyplot(fig, width="stretch", clear_figure=True)

        left, right = st.columns(2)
        with left:
            if st.button("Step"):
                for _ in range(params["steps_per_tick"]):
                    step_sim(state, params)
                st.session_state.state = state
                rerun()
        with right:
            pending_regen = regen_params != st.session_state.applied_regen_params
            if pending_regen:
                st.markdown(
                    "<style>button[data-testid=\"baseButton-primary\"]{box-shadow:0 0 0.65rem #f6b26b;}</style>",
                    unsafe_allow_html=True,
                )
            button_col, note_col = st.columns([1, 2])
            with button_col:
                if st.button(
                    "Regenerate World",
                    type="primary" if pending_regen else "secondary",
                    key="regen_world",
                ):
                    st.session_state.applied_regen_params = copy.deepcopy(regen_params)
                    st.session_state.pop("state", None)
                    rerun()
            with note_col:
                st.caption("Sidebar changes apply after regenerating the world.")

        st.subheader("Playback")
        def toggle_auto_step() -> None:
            st.session_state.auto_step = not st.session_state.auto_step

        run_label = "Stop" if st.session_state.auto_step else "Start"
        st.button(run_label, on_click=toggle_auto_step, key="auto_step_toggle")
        auto_delay_ms = st.slider(
            "Auto-step delay (ms)", 50, 1000, 150, 50, help="Delay between auto steps."
        )

    state = st.session_state.state
    with col2:
        summary = summarize_population(state["agents"])
        prey_traits = summarize_traits(state["agents"], params["metabolism_costs"])
        predator_traits = summarize_traits(state["predators"], params["metabolism_costs"])
        st.metric("Step", state["step"])
        st.metric("Population", summary["pop"])
        st.metric("Predators", len(state["predators"]))
        st.metric("Avg speed", f"{summary['speed']:.2f}")
        st.metric("Avg vision", f"{summary['vision']:.2f}")
        st.metric("Avg metabolism", f"{summary['metabolism']:.2f}")

        if not state["history"] or state["history"][-1]["step"] != state["step"]:
            state["history"].append(
                {
                    "step": state["step"],
                    "pop": summary["pop"],
                    "predators": len(state["predators"]),
                    "speed": summary["speed"],
                    "vision": summary["vision"],
                    "metabolism": summary["metabolism"],
                    "prey_speed": prey_traits["speed"],
                    "prey_vision": prey_traits["vision"],
                    "prey_eat_rate": prey_traits["eat_rate"],
                    "pred_speed": predator_traits["speed"],
                    "pred_vision": predator_traits["vision"],
                    "pred_eat_rate": predator_traits["eat_rate"],
                }
            )
        st.line_chart(
            {
                "Prey": [h["pop"] for h in state["history"]],
                "Predators": [h["predators"] for h in state["history"]],
            },
            height=240,
        )
        st.divider()
        series_options = [
            "Prey speed",
            "Prey vision",
            "Prey eat",
            "Pred speed",
            "Pred vision",
            "Pred eat",
        ]
        selected_series = st.multiselect(
            "Trait series",
            series_options,
            default=["Pred speed", "Pred vision", "Prey vision"],
            help="Choose which trait lines to display.",
        )
        if not selected_series:
            st.info("Select at least one series to display the trait chart.")
        else:
            series_keys = {
                "Prey speed": "prey_speed",
                "Prey vision": "prey_vision",
                "Prey eat": "prey_eat_rate",
                "Pred speed": "pred_speed",
                "Pred vision": "pred_vision",
                "Pred eat": "pred_eat_rate",
            }
            filtered_keys = {k: series_keys[k] for k in selected_series}
            trait_chart = build_trait_chart(state["history"], filtered_keys).properties(
                height=320
            )
            st.altair_chart(trait_chart, width="stretch")

    if st.session_state.auto_step:
        for _ in range(params["steps_per_tick"]):
            step_sim(state, params)
        st.session_state.state = state
        if hasattr(st, "autorefresh"):
            st.autorefresh(interval=auto_delay_ms, key="auto_refresh")
        else:
            time.sleep(auto_delay_ms / 1000.0)
            rerun()

    st.caption(
        "Agents chase food, spend energy to live, and reproduce with mutation when energy is high."
    )


if __name__ == "__main__":
    main()
