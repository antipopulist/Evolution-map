from dataclasses import dataclass

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


def hex_to_rgb(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip("#")
    return np.array(
        [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)], dtype=float
    ) / 255.0


def init_map(size: int, rng: np.random.Generator, terrain_probs: dict) -> dict:
    terrain_ids = np.array(list(terrain_probs.keys()))
    probs = np.array(list(terrain_probs.values()), dtype=float)
    probs = probs / probs.sum()
    terrain = rng.choice(terrain_ids, size=(size, size), p=probs)

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
    count: int, energy: float, rng: np.random.Generator, passable: np.ndarray
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
                "metabolism": float(rng.uniform(0.3, 0.8)),
            }
        )
    return agents


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


def mutate(agent: dict, rng: np.random.Generator, rate: float) -> dict:
    child = agent.copy()
    if rng.random() < rate:
        child["speed"] = int(np.clip(child["speed"] + rng.choice([-1, 1]), 1, 4))
    if rng.random() < rate:
        child["vision"] = int(np.clip(child["vision"] + rng.choice([-1, 1]), 1, 8))
    if rng.random() < rate:
        child["metabolism"] = float(
            np.clip(child["metabolism"] + rng.normal(0, 0.08), 0.1, 1.2)
        )
    return child


def step_sim(state: dict, params: dict) -> dict:
    world = state["world"]
    agents = state["agents"]
    rng = state["rng"]

    world["resource"] = np.minimum(
        world["resource"] + world["regen"] * params["regen_scale"],
        world["max_resource"],
    )

    new_agents = []
    offspring = []
    next_id = state["next_id"]

    for agent in agents:
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
        eaten = min(available, params["eat_rate"])
        world["resource"][nx, ny] -= eaten
        agent["energy"] += eaten * params["energy_gain"]
        agent["energy"] -= agent["metabolism"]

        if agent["energy"] >= params["repro_threshold"]:
            agent["energy"] *= 0.5
            child = mutate(agent, rng, params["mutation_rate"])
            child["id"] = next_id
            next_id += 1
            child["energy"] = agent["energy"]
            offspring.append(child)

        if agent["energy"] > 0:
            new_agents.append(agent)

    state["agents"] = new_agents + offspring
    state["next_id"] = next_id
    state["step"] += 1
    return state


def render_map(world: dict, agents: list) -> plt.Figure:
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


def ensure_state(params: dict) -> None:
    if "state" in st.session_state:
        return
    rng = np.random.default_rng(params["seed"])
    world = init_map(params["size"], rng, params["terrain_probs"])
    agents = init_agents(params["initial_pop"], params["initial_energy"], rng, world["passable"])
    st.session_state.state = {
        "world": world,
        "agents": agents,
        "step": 0,
        "next_id": len(agents),
        "rng": rng,
        "history": [],
    }


def rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def main() -> None:
    st.set_page_config(page_title="Evolution Map", layout="wide")
    st.title("Evolution Map")

    with st.sidebar:
        st.header("World")
        size = st.slider("Map size", 20, 80, 50, 5)
        seed = st.number_input("Seed", value=42, step=1)
        initial_pop = st.slider("Initial population", 10, 200, 80, 10)
        initial_energy = st.slider("Initial energy", 2.0, 10.0, 5.0, 0.5)
        regen_scale = st.slider("Resource regen scale", 0.3, 2.0, 1.0, 0.1)

        st.header("Evolution")
        repro_threshold = st.slider("Reproduction threshold", 4.0, 12.0, 8.0, 0.5)
        mutation_rate = st.slider("Mutation rate", 0.0, 0.5, 0.15, 0.01)
        eat_rate = st.slider("Eat per step", 0.5, 2.0, 1.0, 0.1)
        energy_gain = st.slider("Energy gain per food", 0.5, 3.0, 2.0, 0.1)
        steps_per_tick = st.slider("Steps per tick", 1, 20, 5, 1)

        st.header("Terrain mix")
        plains = st.slider("Plains", 0.1, 0.9, 0.5, 0.05)
        forest = st.slider("Forest", 0.05, 0.7, 0.25, 0.05)
        water = st.slider("Water", 0.0, 0.5, 0.15, 0.05)
        mountain = st.slider("Mountain", 0.0, 0.5, 0.10, 0.05)

    terrain_probs = {0: plains, 1: forest, 2: water, 3: mountain}

    params = {
        "size": size,
        "seed": seed,
        "initial_pop": initial_pop,
        "initial_energy": initial_energy,
        "regen_scale": regen_scale,
        "repro_threshold": repro_threshold,
        "mutation_rate": mutation_rate,
        "eat_rate": eat_rate,
        "energy_gain": energy_gain,
        "steps_per_tick": steps_per_tick,
        "terrain_probs": terrain_probs,
    }

    ensure_state(params)

    col1, col2 = st.columns([2, 1])
    with col1:
        state = st.session_state.state
        fig = render_map(state["world"], state["agents"])
        st.pyplot(fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            if st.button("Step"):
                for _ in range(params["steps_per_tick"]):
                    step_sim(state, params)
                st.session_state.state = state
                rerun()
        with right:
            if st.button("Reset"):
                st.session_state.pop("state", None)
                rerun()

    with col2:
        state = st.session_state.state
        summary = summarize_population(state["agents"])
        st.metric("Step", state["step"])
        st.metric("Population", summary["pop"])
        st.metric("Avg speed", f"{summary['speed']:.2f}")
        st.metric("Avg vision", f"{summary['vision']:.2f}")
        st.metric("Avg metabolism", f"{summary['metabolism']:.2f}")

        if not state["history"] or state["history"][-1]["step"] != state["step"]:
            state["history"].append(
                {
                    "step": state["step"],
                    "pop": summary["pop"],
                    "speed": summary["speed"],
                    "vision": summary["vision"],
                    "metabolism": summary["metabolism"],
                }
            )
        st.line_chart(
            np.array([[h["pop"], h["speed"], h["vision"], h["metabolism"]] for h in state["history"]]),
            height=240,
        )

    st.caption(
        "Agents chase food, spend energy to live, and reproduce with mutation when energy is high."
    )


if __name__ == "__main__":
    main()
