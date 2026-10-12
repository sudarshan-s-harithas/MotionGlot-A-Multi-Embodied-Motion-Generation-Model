"""Checkpoint verification and demo output behavior without remote downloads."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from motionglot.checkpoints import validate_bundle
from motionglot.rendering import recover_human_joints


def test_checkpoint_manifest_rejects_modified_content(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"known checkpoint")
    manifest = {"files": [{"path": path.name, "bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert validate_bundle(tmp_path) == manifest
    path.write_bytes(b"altered checkpoint")
    with pytest.raises(ValueError, match="manifest"):
        validate_bundle(tmp_path)


def test_checkpoint_manifest_rejects_directory_escape(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"files": [{"path": "../outside", "bytes": 0, "sha256": ""}]}))
    with pytest.raises(ValueError, match="manifest path"):
        validate_bundle(tmp_path)


def test_human_joint_recovery_preserves_root_motion():
    features = np.zeros((4, 263), dtype=np.float32)
    features[:, 1] = 0.1
    features[:, 3] = 1
    joints = recover_human_joints(features)
    assert joints.shape == (4, 22, 3)
    np.testing.assert_allclose(joints[:, 0, 0], [0, 0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(joints[:, 0, 1], 1)


def test_human_joint_recovery_rejects_invalid_features():
    with pytest.raises(ValueError):
        recover_human_joints(np.zeros((8, 3)))
    with pytest.raises(ValueError):
        recover_human_joints(np.full((8, 263), np.nan))


def test_demo_accepts_numeric_arrays_and_rejects_object_payloads(tmp_path):
    from motionglot.web_demo import validate_upload
    path = tmp_path / "human.npy"
    np.save(path, np.zeros((252, 263), dtype=np.float32))
    assert validate_upload(path, "human") == str(path)
    with pytest.raises(ValueError):
        validate_upload(path, "robot")
    np.save(path, np.array([{"payload": "object"}], dtype=object))
    with pytest.raises(ValueError):
        validate_upload(path, "human")


def test_demo_renders_a_browser_playable_video(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("imageio")
    pytest.importorskip("imageio_ffmpeg")
    from motionglot.rendering import render_human_video
    import imageio.v2 as imageio
    features = np.zeros((4, 263), dtype=np.float32)
    features[:, 3] = 1
    path = Path(render_human_video(features, tmp_path / "motion.mp4"))
    assert path.stat().st_size > 0
    with imageio.get_reader(path) as reader:
        assert reader.get_data(0).shape == (480, 640, 3)


def test_robot_plot_marks_start_end_and_goal():
    pytest.importorskip("matplotlib")
    from motionglot.rendering import robot_figure
    motion = np.zeros((12, 3))
    motion[:, 0] = 0.1
    plot = robot_figure(motion, goal=(1, 1))
    axes = plot.axes[0]
    np.testing.assert_allclose(axes.lines[0].get_xdata()[-1], 1.2)
    assert len(axes.patches) == 1
