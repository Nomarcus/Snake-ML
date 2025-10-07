"""Standalone Snake game with optional Double DQN training for Python's IDLE."""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

try:  # NumPy krävs för Double DQN-implementationen
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - tydligare fel i IDLE
    raise SystemExit(
        "Det här skriptet kräver NumPy. Installera det via 'pip install numpy' "
        "innan du tränar eller använder autopiloten."
    ) from exc

import tkinter as tk

# ---------------------------------------------------------------------------
# Game configuration
# ---------------------------------------------------------------------------
GRID_SIZE = 15  # Width and height in tiles
CELL_SIZE = 32  # Pixels per tile
STEP_DELAY = 120  # Milliseconds between snake moves
START_LENGTH = 3

# Reward/score configuration
FRUIT_REWARD = 10.0
STEP_PENALTY = -1.0
WALL_PENALTY = -10.0
SELF_PENALTY = -10.0

Direction = Tuple[int, int]
Point = Tuple[int, int]

DIRECTIONS: Dict[str, Direction] = {
    "Up": (0, -1),
    "Down": (0, 1),
    "Left": (-1, 0),
    "Right": (1, 0),
}
ACTION_VECTORS: Tuple[Direction, ...] = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)
OPPOSITE: Dict[Direction, Direction] = {
    (0, -1): (0, 1),
    (0, 1): (0, -1),
    (-1, 0): (1, 0),
    (1, 0): (-1, 0),
}
DIRECTION_TO_INDEX: Dict[Direction, int] = {vec: idx for idx, vec in enumerate(ACTION_VECTORS)}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def grid_size_state_size(grid_size: int) -> int:
    return grid_size * grid_size * 2 + len(ACTION_VECTORS)


def build_state_vector(
    snake: Iterable[Point],
    fruit: Point,
    direction_index: int,
    grid_size: int,
) -> np.ndarray:
    """Flattened observation used both for training and the Tk autopilot."""

    snake_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    fruit_grid = np.zeros_like(snake_grid)
    for x, y in snake:
        if 0 <= x < grid_size and 0 <= y < grid_size:
            snake_grid[y, x] = 1.0
    fx, fy = fruit
    if 0 <= fx < grid_size and 0 <= fy < grid_size:
        fruit_grid[fy, fx] = 1.0
    direction_one_hot = np.zeros(len(ACTION_VECTORS), dtype=np.float32)
    direction_one_hot[direction_index] = 1.0
    return np.concatenate((snake_grid.flatten(), fruit_grid.flatten(), direction_one_hot))


# ---------------------------------------------------------------------------
# Double DQN training components
# ---------------------------------------------------------------------------


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Basic replay buffer for off-policy training."""

    def __init__(self, capacity: int, rng: random.Random) -> None:
        self.capacity = capacity
        self.memory: Deque[Transition] = deque(maxlen=capacity)
        self.rng = rng

    def add(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> Transition:
        batch = self.rng.sample(self.memory, batch_size)
        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return Transition(states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        return len(self.memory)


class MLP:
    """Small fully connected network implemented with NumPy."""

    def __init__(self, layer_sizes: List[int], seed: Optional[int] = None) -> None:
        self.layer_sizes = layer_sizes
        self.rng = np.random.default_rng(seed)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = math.sqrt(6.0 / (in_size + out_size))
            weight = self.rng.uniform(-limit, limit, size=(in_size, out_size)).astype(np.float32)
            bias = np.zeros(out_size, dtype=np.float32)
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, inputs: np.ndarray, return_cache: bool = False):
        x = inputs.astype(np.float32)
        activations = [x]
        pre_activations = []
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = x @ weight + bias
            pre_activations.append(x)
            if index < len(self.weights) - 1:
                x = np.maximum(x, 0.0)
            activations.append(x)
        if return_cache:
            return x, (activations, pre_activations)
        return x

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs, return_cache=False)

    def backward(self, grad_output: np.ndarray, cache, learning_rate: float) -> None:
        activations, pre_activations = cache
        grad = grad_output
        for layer in reversed(range(len(self.weights))):
            a_prev = activations[layer]
            grad_w = a_prev.T @ grad
            grad_b = grad.sum(axis=0)
            self.weights[layer] -= learning_rate * grad_w
            self.biases[layer] -= learning_rate * grad_b
            if layer > 0:
                grad = grad @ self.weights[layer].T
                grad = grad * (pre_activations[layer - 1] > 0)

    def copy(self) -> "MLP":
        clone = MLP(self.layer_sizes)
        clone.weights = [weight.copy() for weight in self.weights]
        clone.biases = [bias.copy() for bias in self.biases]
        return clone


class DoubleDQNAgent:
    """Double DQN agent implemented specifically for the IDLE snake grid."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        *,
        hidden_layers: Optional[List[int]] = None,
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 50_000,
        target_sync_interval: int = 500,
        seed: Optional[int] = None,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layers = hidden_layers or [128, 128]
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_sync_interval = target_sync_interval
        self.train_steps = 0

        rng_seed = seed if seed is not None else random.randrange(1_000_000)
        self.random = random.Random(rng_seed)
        self.layer_sizes = [state_size, *self.hidden_layers, action_size]
        self.online = MLP(self.layer_sizes, seed=rng_seed)
        self.target = self.online.copy()
        self.buffer = ReplayBuffer(buffer_capacity, self.random)

    # ------------------------------------------------------------------
    # Persistence utilities
    # ------------------------------------------------------------------
    def save(self, path: Path | str) -> Path:
        path = Path(path)
        arrays = {
            "layer_sizes": np.array(self.layer_sizes, dtype=np.int64),
            "learning_rate": np.array([self.learning_rate], dtype=np.float32),
            "gamma": np.array([self.gamma], dtype=np.float32),
            "epsilon": np.array([self.epsilon], dtype=np.float32),
            "epsilon_min": np.array([self.epsilon_min], dtype=np.float32),
            "epsilon_decay": np.array([self.epsilon_decay], dtype=np.float32),
            "batch_size": np.array([self.batch_size], dtype=np.int64),
            "target_sync": np.array([self.target_sync_interval], dtype=np.int64),
        }
        for idx, weight in enumerate(self.online.weights):
            arrays[f"W{idx}"] = weight
        for idx, bias in enumerate(self.online.biases):
            arrays[f"b{idx}"] = bias
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "DoubleDQNAgent":
        data = np.load(Path(path), allow_pickle=False)
        layer_sizes = data["layer_sizes"].astype(np.int64).tolist()
        state_size = int(layer_sizes[0])
        action_size = int(layer_sizes[-1])
        hidden_layers = [int(x) for x in layer_sizes[1:-1]]
        agent = cls(
            state_size,
            action_size,
            hidden_layers=hidden_layers,
            learning_rate=float(data["learning_rate"][0]),
            gamma=float(data["gamma"][0]),
            epsilon_start=float(data["epsilon"][0]),
            epsilon_end=float(data["epsilon_min"][0]),
            epsilon_decay=float(data["epsilon_decay"][0]),
            batch_size=int(data["batch_size"][0]),
            target_sync_interval=int(data["target_sync"][0]),
        )
        for idx in range(len(agent.online.weights)):
            agent.online.weights[idx] = data[f"W{idx}"]
            agent.online.biases[idx] = data[f"b{idx}"]
        agent.target = agent.online.copy()
        return agent

    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if state.ndim == 1:
            state = state[None, :]
        if not greedy and self.random.random() < self.epsilon:
            return self.random.randrange(self.action_size)
        q_values = self.online.predict(state)
        if q_values.ndim == 2:
            q_values = q_values[0]
        return int(np.argmax(q_values))

    def push(self, transition: Transition) -> None:
        self.buffer.add(transition)

    def decay_epsilon(self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self) -> None:
        self.target = self.online.copy()

    def learn(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.buffer.sample(self.batch_size)
        states = batch.state
        actions = batch.action
        rewards = batch.reward
        next_states = batch.next_state
        dones = batch.done

        q_values, cache = self.online.forward(states, return_cache=True)
        q_next_online = self.online.predict(next_states)
        q_next_target = self.target.predict(next_states)
        next_actions = np.argmax(q_next_online, axis=1)
        targets = rewards + (1.0 - dones) * self.gamma * q_next_target[np.arange(self.batch_size), next_actions]

        diff = q_values[np.arange(self.batch_size), actions] - targets
        loss = float(np.mean(diff ** 2) * 0.5)
        grad_output = np.zeros_like(q_values)
        grad_output[np.arange(self.batch_size), actions] = diff / self.batch_size
        self.online.backward(grad_output, cache, self.learning_rate)

        self.train_steps += 1
        if self.train_steps % self.target_sync_interval == 0:
            self.update_target()
        return loss


class IdleSnakeEnv:
    """Lightweight snake environment for headless Double DQN training."""

    def __init__(self, grid_size: int = GRID_SIZE, seed: Optional[int] = None) -> None:
        self.grid_size = grid_size
        self.random = random.Random(seed)
        self.snake: List[Point] = []
        self.direction_index: int = 1
        self.fruit: Point = (0, 0)
        self.pending_growth = 0
        self.state = np.zeros(grid_size_state_size(grid_size), dtype=np.float32)

    @property
    def state_size(self) -> int:
        return grid_size_state_size(self.grid_size)

    def reset(self) -> np.ndarray:
        start_x = self.grid_size // 2
        start_y = self.grid_size // 2
        self.snake = [(start_x - i, start_y) for i in range(START_LENGTH)]
        self.direction_index = 1
        self.pending_growth = 0
        self._spawn_fruit()
        self.state = build_state_vector(self.snake, self.fruit, self.direction_index, self.grid_size)
        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, str]]:
        if action < 0 or action >= len(ACTION_VECTORS):
            raise ValueError("Action out of range")
        current_direction = ACTION_VECTORS[self.direction_index]
        chosen_direction = ACTION_VECTORS[action]
        if chosen_direction == OPPOSITE[current_direction]:
            chosen_direction = current_direction
            action = self.direction_index
        self.direction_index = DIRECTION_TO_INDEX[chosen_direction]

        head_x, head_y = self.snake[0]
        dx, dy = chosen_direction
        new_head = (head_x + dx, head_y + dy)

        reward = STEP_PENALTY
        done = False
        cause = "step"

        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            reward += WALL_PENALTY + SELF_PENALTY
            done = True
            cause = "wall"
        elif new_head in self.snake:
            reward += SELF_PENALTY
            done = True
            cause = "self"
        else:
            self.snake.insert(0, new_head)
            if new_head == self.fruit:
                reward += FRUIT_REWARD
                self.pending_growth += 1
                self._spawn_fruit()
                cause = "fruit"
            if self.pending_growth > 0:
                self.pending_growth -= 1
            else:
                self.snake.pop()

        if done:
            next_state = self.state.copy()
        else:
            next_state = build_state_vector(self.snake, self.fruit, self.direction_index, self.grid_size)
            self.state = next_state.copy()
        return next_state, reward, done, {"cause": cause}

    def _spawn_fruit(self) -> None:
        free_cells = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in self.snake
        ]
        self.fruit = self.random.choice(free_cells) if free_cells else self.snake[0]

    def snapshot(self) -> Dict[str, object]:
        """Return a shallow copy of the current game state for visualisering."""

        return {
            "snake": list(self.snake),
            "fruit": self.fruit,
            "direction_index": self.direction_index,
            "pending_growth": self.pending_growth,
        }


# ---------------------------------------------------------------------------
# Tkinter score tracking and rendering
# ---------------------------------------------------------------------------


@dataclass
class Score:
    fruits: int = 0
    steps: int = 0
    reward: float = 0.0

    def reset(self) -> None:
        self.fruits = 0
        self.steps = 0
        self.reward = 0.0

    def apply_step(self, outcome: str) -> None:
        self.steps += 1
        self.reward += STEP_PENALTY
        if outcome == "fruit":
            self.fruits += 1
            self.reward += FRUIT_REWARD - STEP_PENALTY
        elif outcome == "wall":
            self.reward += WALL_PENALTY
        elif outcome == "self":
            self.reward += SELF_PENALTY


class SnakeCanvas(tk.Canvas):
    """Canvas widget that hosts the actual snake game."""

    def __init__(self, master: tk.Misc, status: tk.Label, agent: Optional[DoubleDQNAgent] = None, autopilot: bool = False) -> None:
        pixel_size = GRID_SIZE * CELL_SIZE
        super().__init__(
            master,
            width=pixel_size,
            height=pixel_size,
            bg="#151515",
            highlightthickness=0,
        )
        self.pack()
        self.status_label = status
        self.score = Score()
        self._running = False
        self._after_id: Optional[str] = None
        self.agent = agent
        self.autopilot = autopilot and agent is not None
        self.reset()
        self.focus_set()
        self.bind("<KeyPress>", self.on_key_press)

    def reset(self) -> None:
        self.delete("all")
        start_x = GRID_SIZE // 2
        start_y = GRID_SIZE // 2
        self.snake = [(start_x - i, start_y) for i in range(START_LENGTH)]
        self.direction = (1, 0)
        self.score.reset()
        self._running = True
        self.spawn_fruit()
        self.draw_frame()
        self.update_status("start")

    def spawn_fruit(self) -> None:
        free_cells = [
            (x, y)
            for x in range(GRID_SIZE)
            for y in range(GRID_SIZE)
            if (x, y) not in self.snake
        ]
        self.fruit = random.choice(free_cells) if free_cells else self.snake[0]

    # Input handling -----------------------------------------------------
    def on_key_press(self, event: tk.Event[tk.Misc]) -> None:
        if event.keysym == "space":
            self.reset()
            return
        if event.keysym == "Escape":
            self.quit_game()
            return
        if event.keysym.lower() == "a":
            self.toggle_autopilot()
            return
        if event.keysym not in DIRECTIONS:
            return
        new_dir = DIRECTIONS[event.keysym]
        if OPPOSITE.get(self.direction) == new_dir:
            return
        self.direction = new_dir

    def toggle_autopilot(self) -> None:
        if self.agent is None:
            print("Ingen tränad agent laddad. Använd --load-model för att aktivera autopilot.")
            return
        self.autopilot = not self.autopilot
        mode = "på" if self.autopilot else "av"
        print(f"Autopilot är nu {mode}.")
        self.update_status("step")

    def quit_game(self) -> None:
        self._running = False
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        self.master.destroy()

    # Rendering helpers --------------------------------------------------
    def draw_frame(self) -> None:
        self.delete("all")
        self.draw_grid()
        self.draw_snake()
        self.draw_fruit()

    def draw_grid(self) -> None:
        pixel_size = GRID_SIZE * CELL_SIZE
        for row in range(GRID_SIZE):
            color = "#1f1f1f" if row % 2 == 0 else "#232323"
            self.create_rectangle(
                0,
                row * CELL_SIZE,
                pixel_size,
                (row + 1) * CELL_SIZE,
                fill=color,
                outline=color,
            )

    def draw_snake(self) -> None:
        for index, (x, y) in enumerate(self.snake):
            fill = "#32c85c" if index == 0 else "#58e279"
            self._draw_cell(x, y, fill)

    def draw_fruit(self) -> None:
        fx, fy = self.fruit
        self._draw_cell(fx, fy, "#ff5c5c")

    def _draw_cell(self, x: int, y: int, color: str) -> None:
        self.create_rectangle(
            x * CELL_SIZE + 2,
            y * CELL_SIZE + 2,
            (x + 1) * CELL_SIZE - 2,
            (y + 1) * CELL_SIZE - 2,
            fill=color,
            outline="",
        )

    # Game loop ----------------------------------------------------------
    def start(self) -> None:
        self.after(300, self.tick)

    def tick(self) -> None:
        if not self._running:
            return
        if self.autopilot and self.agent is not None:
            direction_idx = DIRECTION_TO_INDEX[self.direction]
            state = build_state_vector(self.snake, self.fruit, direction_idx, GRID_SIZE)
            action = self.agent.select_action(state, greedy=True)
            chosen_direction = ACTION_VECTORS[action]
            if OPPOSITE[self.direction] != chosen_direction:
                self.direction = chosen_direction
        outcome = self.advance_snake()
        self.draw_frame()
        self.score.apply_step(outcome)
        self.update_status(outcome)
        if outcome in {"wall", "self"}:
            self._running = False
            self.update_status(outcome, finished=True)
            return
        self._after_id = self.after(STEP_DELAY, self.tick)

    def update_status(self, outcome: str, finished: bool = False) -> None:
        if outcome == "fruit":
            prefix = "🍎 Frukt!"
        elif outcome == "wall":
            prefix = "💥 Krockade med väggen."
        elif outcome == "self":
            prefix = "💥 Åt sig själv."
        elif outcome == "start":
            prefix = "🐍 Nystart."
        else:
            prefix = "🐍"
        suffix = " Tryck Space för att börja om." if finished else ""
        autopilot_text = " 🤖" if self.autopilot and self.agent is not None else ""
        self.status_label.configure(
            text=(
                f"{prefix}{autopilot_text} Poäng: {self.score.reward:.0f}  "
                f"Frukter: {self.score.fruits}  Steg: {self.score.steps}.{suffix}"
            )
        )

    def advance_snake(self) -> str:
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        if not (0 <= new_head[0] < GRID_SIZE and 0 <= new_head[1] < GRID_SIZE):
            return "wall"
        if new_head in self.snake:
            return "self"

        self.snake.insert(0, new_head)
        if new_head == self.fruit:
            self.spawn_fruit()
            return "fruit"
        self.snake.pop()
        return "step"


class TrainingViewer:
    """Lightweight Tkinter-visualisering av träningsmiljön."""

    def __init__(
        self,
        *,
        grid_size: int,
        total_episodes: int,
        steps_per_episode: int,
        cell_size: int = CELL_SIZE,
        delay_ms: int = STEP_DELAY,
    ) -> None:
        self.grid_size = grid_size
        self.total_episodes = total_episodes
        self.steps_per_episode = steps_per_episode
        self.cell_size = cell_size
        self.delay = max(delay_ms / 1000.0, 0.0)
        self._last_draw = 0.0
        self.closed = False

        self.root = tk.Tk()
        self.root.title("Snake-ML – Träningsvisualisering")
        self.root.configure(bg="#101010")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.status = tk.Label(
            self.root,
            text="Initierar träning…",
            font=("Segoe UI", 12),
            anchor="w",
            padx=12,
            pady=8,
            bg="#101010",
            fg="#f5f5f5",
        )
        self.status.pack(fill="x")

        pixel_size = self.grid_size * self.cell_size
        self.canvas = tk.Canvas(
            self.root,
            width=pixel_size,
            height=pixel_size,
            bg="#151515",
            highlightthickness=0,
        )
        self.canvas.pack()

        self._draw_background()
        self.root.update_idletasks()
        self.root.update()

    def update(
        self,
        *,
        snapshot: Dict[str, object],
        episode: int,
        step: int,
        epsilon: float,
        total_reward: float,
        last_reward: float,
        cause: str,
        loss: float,
    ) -> None:
        if self.closed:
            return
        now = time.perf_counter()
        if now - self._last_draw < self.delay:
            self.root.update_idletasks()
            self.root.update()
            return
        self._last_draw = now

        snake = list(snapshot.get("snake", []))
        fruit = snapshot.get("fruit")

        self.canvas.delete("all")
        self._draw_background()
        self._draw_fruit(fruit)
        self._draw_snake(snake)

        status_text = (
            f"Episod {episode}/{self.total_episodes} | Steg {step}/{self.steps_per_episode} | "
            f"Reward {total_reward:6.1f} | Δ {last_reward:5.2f} | ε={epsilon:.3f} | "
            f"Senaste: {self._format_cause(cause)} | Förlust {loss:7.4f}"
        )
        self.status.config(text=status_text)

        self.root.update_idletasks()
        self.root.update()

    def episode_done(self) -> None:
        if self.closed:
            return
        current = self.status.cget("text")
        if "| Slut" not in current:
            self.status.config(text=f"{current} | Slut på episod")
        self.root.update_idletasks()
        self.root.update()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _draw_background(self) -> None:
        pixel_size = self.grid_size * self.cell_size
        for row in range(self.grid_size):
            color = "#1f1f1f" if row % 2 == 0 else "#232323"
            self.canvas.create_rectangle(
                0,
                row * self.cell_size,
                pixel_size,
                (row + 1) * self.cell_size,
                fill=color,
                width=0,
            )

    def _draw_snake(self, snake: List[Point]) -> None:
        for index, (x, y) in enumerate(snake):
            color = "#8bc34a" if index == 0 else "#4caf50"
            self.canvas.create_rectangle(
                x * self.cell_size,
                y * self.cell_size,
                (x + 1) * self.cell_size,
                (y + 1) * self.cell_size,
                fill=color,
                outline="#1b5e20",
                width=1,
            )

    def _draw_fruit(self, fruit: Optional[Point]) -> None:
        if fruit is None:
            return
        x, y = fruit
        self.canvas.create_oval(
            x * self.cell_size + 4,
            y * self.cell_size + 4,
            (x + 1) * self.cell_size - 4,
            (y + 1) * self.cell_size - 4,
            fill="#ff9800",
            outline="#ef6c00",
            width=1,
        )

    def _format_cause(self, cause: str) -> str:
        translations = {
            "fruit": "frukt",
            "wall": "vägg",
            "self": "kollision",
            "step": "steg",
        }
        return translations.get(cause, cause)


# ---------------------------------------------------------------------------
# Training / evaluation CLI
# ---------------------------------------------------------------------------


def train_double_dqn(
    episodes: int,
    steps_per_episode: int,
    *,
    seed: Optional[int] = None,
    load_path: Optional[Path | str] = None,
    save_path: Optional[Path | str] = None,
    visualize: bool = False,
) -> DoubleDQNAgent:
    env = IdleSnakeEnv(seed=seed)
    if load_path is not None:
        agent = DoubleDQNAgent.load(load_path)
        print(f"Fortsätter träning från {load_path}.")
    else:
        agent = DoubleDQNAgent(state_size=env.state_size, action_size=len(ACTION_VECTORS), seed=seed)
        print("Startar ny Double DQN-träning.")

    rewards: List[float] = []
    start_time = time.time()
    viewer: Optional[TrainingViewer] = None
    if visualize:
        try:
            viewer = TrainingViewer(
                grid_size=env.grid_size,
                total_episodes=episodes,
                steps_per_episode=steps_per_episode,
            )
        except tk.TclError as exc:
            print(
                "Det gick inte att starta träningsvisualiseringen (Tkinter). Fortsätter utan grafiskt läge.",
                file=sys.stderr,
            )
            viewer = None

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        losses: List[float] = []
        for step in range(steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.push(Transition(state, action, reward, next_state, done))
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            agent.decay_epsilon()
            state = next_state
            total_reward += reward
            if viewer is not None:
                try:
                    viewer.update(
                        snapshot=env.snapshot(),
                        episode=episode,
                        step=step + 1,
                        epsilon=agent.epsilon,
                        total_reward=total_reward,
                        last_reward=reward,
                        cause=info.get("cause", "step"),
                        loss=losses[-1] if losses else 0.0,
                    )
                except tk.TclError:
                    print("Träningsfönstret stängdes – fortsätter träningen utan visualisering.")
                    viewer = None
            if done:
                break
        rewards.append(total_reward)
        if episode % 10 == 0:
            mean_reward = sum(rewards[-10:]) / min(len(rewards), 10)
            mean_loss = sum(losses) / len(losses) if losses else 0.0
            duration = time.time() - start_time
            print(
                f"Episod {episode:4d} | Snittbelöning (10): {mean_reward:6.2f} | "
                f"Senaste episod: {total_reward:6.2f} | Förlust: {mean_loss:8.4f} | "
                f"ε={agent.epsilon:.3f} | Tid: {duration:5.1f}s"
            )
        if viewer is not None:
            try:
                viewer.episode_done()
            except tk.TclError:
                print("Träningsfönstret stängdes – fortsätter träningen utan visualisering.")
                viewer = None

    if save_path is not None:
        path = agent.save(save_path)
        print(f"Sparade modellen till {path}.")
    if viewer is not None:
        viewer.close()
    return agent


def evaluate_agent(agent: DoubleDQNAgent, episodes: int, steps_per_episode: int, *, seed: Optional[int] = None) -> List[float]:
    env = IdleSnakeEnv(seed=seed)
    results: List[float] = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        for _ in range(steps_per_episode):
            action = agent.select_action(state, greedy=True)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        results.append(total_reward)
    return results


def start_game(agent: Optional[DoubleDQNAgent] = None, autopilot: bool = False) -> None:
    root = tk.Tk()
    root.title("Snake-ML – IDLE Edition")
    root.resizable(False, False)
    root.configure(bg="#101010")

    status = tk.Label(
        root,
        text="Tryck på piltangenterna för att styra. Space startar om, Escape avslutar.",
        font=("Segoe UI", 12),
        bg="#101010",
        fg="#f5f5f5",
        anchor="w",
        padx=12,
        pady=8,
    )
    status.pack(fill="x")

    canvas = SnakeCanvas(root, status, agent=agent, autopilot=autopilot)
    canvas.pack()
    canvas.start()

    root.mainloop()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snake-ML IDLE Double DQN toolkit")
    parser.add_argument("--train", type=int, metavar="EP", help="Antal träningsepisoder att köra", default=0)
    parser.add_argument("--steps", type=int, metavar="N", help="Maxsteg per episod under träning", default=500)
    parser.add_argument("--seed", type=int, help="Slumpfrö", default=None)
    parser.add_argument("--load-model", type=str, help="Ladda en sparad agent för träning eller spel", default=None)
    parser.add_argument("--save-model", type=str, help="Var ska modellen sparas efter träning", default="idle_double_dqn.npz")
    parser.add_argument("--evaluate", type=int, metavar="EP", help="Kör utvärdering med angivet antal episoder", default=0)
    parser.add_argument("--play", action="store_true", help="Starta Tkinter-spelet")
    parser.add_argument("--autopilot", action="store_true", help="Aktivera autopilot när spelet startas")
    parser.add_argument(
        "--visualize-training",
        action="store_true",
        help="Visa träningsmiljön live i ett Tkinter-fönster",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    agent: Optional[DoubleDQNAgent] = None
    if args.train > 0:
        agent = train_double_dqn(
            args.train,
            args.steps,
            seed=args.seed,
            load_path=args.load_model,
            save_path=args.save_model,
            visualize=args.visualize_training,
        )
    elif args.load_model:
        agent = DoubleDQNAgent.load(args.load_model)
        print(f"Laddade agent från {args.load_model}.")

    if args.evaluate:
        if agent is None:
            raise SystemExit("Ingen agent att utvärdera. Träna först eller ladda en modell.")
        scores = evaluate_agent(agent, args.evaluate, args.steps, seed=args.seed)
        avg = sum(scores) / len(scores)
        best = max(scores)
        worst = min(scores)
        print(
            f"Utvärdering över {args.evaluate} episoder | Snitt: {avg:.2f} | "
            f"Bäst: {best:.2f} | Sämst: {worst:.2f}"
        )

    if args.play or (agent and args.autopilot):
        try:
            start_game(agent=agent, autopilot=args.autopilot)
        except tk.TclError as exc:  # pragma: no cover - headless safeguard
            print(
                "Det gick inte att starta Tkinter. Om du kör skriptet på en server utan skärm",
                "behöver du köra det lokalt i stället.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc
    elif args.train == 0 and args.evaluate == 0:
        # Default behaviour when running without flags.
        try:
            start_game(agent=None, autopilot=False)
        except tk.TclError as exc:  # pragma: no cover - headless safeguard
            print(
                "Det gick inte att starta Tkinter. Om du kör skriptet på en server utan skärm",
                "behöver du köra det lokalt i stället.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
