"""The six tasks and exact token spellings recovered from MotionGlot."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Task:
    prefix: str
    start: str
    end: str
    output: str


TASKS = {
    "text-to-human": Task("give human motion: ", "<toh>", "<eog_tthm>", "human"),
    "question-to-human": Task("give human motion: ", "<toh>", "<eog_tthm>", "human"),
    "text-to-robot": Task("give robot motion: ", "<tor>", "<eog_ttrm>", "robot"),
    "human-to-text": Task("give text description: ", "<mtt>", "<eog_hmtt>", "text"),
    "robot-to-text": Task("give text description: ", "<rmtt>", "<eog_rmtt>", "text"),
    # The recovered checkpoint was trained without the paper's 'reach goal: ' prefix.
    "goal-to-robot": Task("", "<tog>", "<eog_rmg>", "robot"),
}
SPECIAL_TOKENS = ["<eop>", "<toh>", "<tor>", "<tog>", "<mtt>", "<rmtt>",
                  "<trott>", "<gallop>", "<eog_tthm>", "<eog_ttrm>",
                  "<eog_hmtt>", "<eog_rmtt>", "<eog_rmg>"]


def motion_string(codes, embodiment):
    if embodiment not in ("human", "robot"):
        raise ValueError("Unknown embodiment")
    values = [int(i) for i in codes]
    if not values or any(i < 0 for i in values):
        raise ValueError("Motion needs nonnegative code indices")
    return "".join("<{}_motion_id_{}>".format(embodiment, i) for i in values)


def prompt(task, source):
    spec = TASKS[task]
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Task input cannot be empty")
    return spec.prefix + source + "<eop>" + spec.start + (" \n " if spec.output != "text" else "")


def instruction(task, source, target, gait=None):
    if not isinstance(target, str) or not target:
        raise ValueError("Task target cannot be empty")
    if gait is not None and (task != "text-to-robot" or gait not in ("trott", "gallop")):
        raise ValueError("Gaits trott/gallop apply to text-to-robot")
    return prompt(task, source) + ("<{}>".format(gait) if gait else "") + target + TASKS[task].end


def goal_id(x, y, minimum=-7.0, maximum=7.0, cell_size=0.5):
    if not all(math.isfinite(v) for v in (x, y)) or not (minimum <= x < maximum and minimum <= y < maximum):
        raise ValueError("Goal is outside the [-7,7) metre grid")
    n = int(round((maximum - minimum) / cell_size))
    # x increases left-to-right; y centres descend, matching the saved vocabulary.
    ix = int(math.floor((x - minimum) / cell_size))
    iy = n - 1 - int(math.floor((y - minimum) / cell_size))
    return ix * n + iy


def goal_center(index):
    if not 0 <= index < 784:
        raise ValueError("Goal index must be in [0,783]")
    ix, iy = divmod(index, 28)
    return -6.75 + ix * 0.5, 6.75 - iy * 0.5
