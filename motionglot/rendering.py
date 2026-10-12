"""HumanML3D recovery and portable MP4/trajectory visualization.

Human feature recovery is adapted from the recovered MotionGlot post_process.py
and HumanML3D utilities. See THIRD_PARTY_NOTICES.md for attribution.
"""
import math
from pathlib import Path
import numpy as np
import torch

CHAINS = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
          [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
COLORS = ["#bd3c5b", "#4169ad", "#283b4e", "#bd3c5b", "#4169ad"]


def _qrot(q, v):
    qvector = q[..., 1:]
    cross = torch.cross(qvector, v, dim=-1)
    cross2 = torch.cross(qvector, cross, dim=-1)
    return v + 2 * (q[..., :1] * cross + cross2)


def recover_human_joints(features):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 263 or not np.isfinite(features).all() or not len(features):
        raise ValueError("Human features must be finite [frames,263] data")
    data = torch.from_numpy(features.copy())
    angles = torch.zeros(len(data))
    angles[1:] = data[:-1, 0]
    angles = angles.cumsum(0)
    q = torch.zeros(len(data), 4)
    q[:, 0], q[:, 2] = angles.cos(), angles.sin()
    inverse = q.clone()
    inverse[:, 1:] *= -1
    root = torch.zeros(len(data), 3)
    root[1:, [0, 2]] = data[:-1, 1:3]
    root = _qrot(inverse, root).cumsum(0)
    root[:, 1] = data[:, 3]
    joints = data[:, 4:67].reshape(len(data), 21, 3)
    joints = _qrot(inverse[:, None, :].expand(-1, 21, -1), joints)
    joints[:, :, 0] += root[:, 0:1]
    joints[:, :, 2] += root[:, 2:3]
    return torch.cat([root[:, None, :], joints], dim=1).numpy()


def render_human_video(features, output, fps=20, max_frames=160):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    joints = recover_human_joints(features)
    joints[..., 1] -= joints[..., 1].min()
    stride = max(1, math.ceil(len(joints) / max_frames))
    frames = joints[::stride]
    low, high = frames.min(axis=(0, 1)), frames.max(axis=(0, 1))
    centre = (low + high) / 2
    radius = max(1.0, float((high - low).max()) / 2 + 0.3)
    fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(str(output), fps=fps / stride, codec="libx264",
                                pixelformat="yuv420p", macro_block_size=None) as writer:
            for index, pose in enumerate(frames):
                ax.clear()
                ax.set_xlim(centre[0] - radius, centre[0] + radius)
                ax.set_ylim(centre[2] - radius, centre[2] + radius)
                ax.set_zlim(0, max(2.2, high[1] + 0.2))
                ax.view_init(elev=18, azim=-55)
                ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.set_zlabel("height (m)")
                path = frames[:index + 1, 0]
                ax.plot(path[:, 0], path[:, 2], np.zeros(len(path)), color="#b8bec8", linewidth=1)
                for chain, color in zip(CHAINS, COLORS):
                    ax.plot(pose[chain, 0], pose[chain, 2], pose[chain, 1], color=color, linewidth=3)
                fig.tight_layout()
                fig.canvas.draw()
                writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    finally:
        plt.close(fig)
    return str(output)


def robot_figure(motion, goal=None):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    from .tasks import goal_id, goal_center
    motion = np.asarray(motion)
    if motion.ndim != 2 or motion.shape[1] != 3 or not len(motion) or not np.isfinite(motion).all():
        raise ValueError("Robot motion must be finite [frames,3] data")
    trajectory = np.vstack([np.zeros((1, 3)), motion.cumsum(0)])
    fig = Figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.plot(trajectory[:, 0], trajectory[:, 1], color="#4169ad", linewidth=2, label="Generated trajectory")
    ax.scatter(*trajectory[0, :2], color="#369870", label="Start", zorder=3)
    ax.scatter(*trajectory[-1, :2], color="#bd3c5b", label="End", zorder=3)
    points = np.linspace(0, len(trajectory) - 1, min(8, len(trajectory)), dtype=int)
    ax.quiver(trajectory[points, 0], trajectory[points, 1], np.cos(trajectory[points, 2]),
              np.sin(trajectory[points, 2]), color="#283b4e", scale=15)
    if goal is not None:
        x, y = goal_center(goal_id(*goal))
        ax.add_patch(Rectangle((x - 0.25, y - 0.25), 0.5, 0.5, color="#efca73", alpha=0.5, label="Goal cell"))
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.axis("equal"); ax.grid(alpha=0.2); ax.legend()
    fig.tight_layout()
    return fig
