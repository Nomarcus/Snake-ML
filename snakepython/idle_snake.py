"""Standalone Snake game that works directly in Python's IDLE.

The goal of this script is to offer a tiny, dependency-free way to play Snake
without having to install virtual environments or additional packages. It keeps
roughly the same scoring idea as the larger Snake-ML project but focuses on a
simple Tkinter window that can be launched from IDLE's "Run Module" command.
"""
from __future__ import annotations

import random
import sys
import tkinter as tk
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Game configuration
# ---------------------------------------------------------------------------
GRID_SIZE = 15  # Width and height in tiles
CELL_SIZE = 32  # Pixels per tile
STEP_DELAY = 120  # Milliseconds between snake moves
START_LENGTH = 3

# Score handling similar to the reward setup in the HTML version
FRUIT_REWARD = 10
STEP_PENALTY = -1
WALL_PENALTY = -10
SELF_PENALTY = -10

Direction = Tuple[int, int]
Point = Tuple[int, int]

DIRECTIONS: Dict[str, Direction] = {
    "Up": (0, -1),
    "Down": (0, 1),
    "Left": (-1, 0),
    "Right": (1, 0),
}
OPPOSITE: Dict[Direction, Direction] = {
    (0, -1): (0, 1),
    (0, 1): (0, -1),
    (-1, 0): (1, 0),
    (1, 0): (-1, 0),
}


@dataclass
class Score:
    fruits: int = 0
    steps: int = 0
    reward: int = 0

    def reset(self) -> None:
        self.fruits = 0
        self.steps = 0
        self.reward = 0

    def apply_step(self, outcome: str) -> None:
        """Update counters based on the last action outcome."""

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

    snake: List[Point]
    fruit: Point
    direction: Direction

    def __init__(self, master: tk.Misc, status: tk.Label) -> None:
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
        self._after_id: str | None = None
        self.reset()
        self.focus_set()
        self.bind("<KeyPress>", self.on_key_press)

    # ------------------------------------------------------------------
    # Game setup
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def on_key_press(self, event: tk.Event[tk.Misc]) -> None:
        if event.keysym == "space":
            self.reset()
            return
        if event.keysym == "Escape":
            self.quit_game()
            return
        if event.keysym not in DIRECTIONS:
            return
        new_dir = DIRECTIONS[event.keysym]
        if OPPOSITE.get(self.direction) == new_dir:
            return
        self.direction = new_dir

    def quit_game(self) -> None:
        self._running = False
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        self.master.destroy()

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Game loop
    # ------------------------------------------------------------------
    def start(self) -> None:
        self.after(300, self.tick)

    def tick(self) -> None:
        if not self._running:
            return
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
        self.status_label.configure(
            text=(
                f"{prefix} Poäng: {self.score.reward}  "
                f"Frukter: {self.score.fruits}  Steg: {self.score.steps}."
                f"{suffix}"
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_game() -> None:
    """Launch the Tkinter window and start the game loop."""

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

    canvas = SnakeCanvas(root, status)
    canvas.pack()
    canvas.start()

    root.mainloop()


if __name__ == "__main__":
    try:
        start_game()
    except tk.TclError as exc:  # pragma: no cover - headless safeguard
        print(
            "Det gick inte att starta Tkinter. Om du kör skriptet på en server utan skärm",
            "behöver du köra det lokalt i stället.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
