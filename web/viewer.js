import { createEngine } from "https://cdn.jsdelivr.net/npm/mjswan@0.11.6/dist/mjswan.js";
import {
  zipSync,
  strToU8,
} from "https://cdn.jsdelivr.net/npm/fflate@0.8.2/+esm";
import { bones, humanJoints } from "./motion.js?v=human1";

// Version-pinned bridge to mjswan's runtime. No mj_step or engine.play call.
// The public embedding API currently has no per-frame pose setter. Keep the
// internal bridge isolated here, and test it when updating mjswan.
export class MotionViewer {
  async init(element) {
    this.engine = await createEngine(element, {
      multithreaded: false,
      handTracking: false,
    });
    this.runtime = this.engine.runtime;
    if (
      !this.runtime?.mujoco ||
      typeof this.runtime.updateCachedState !== "function"
    )
      throw Error("The mjswan playback bridge is incompatible.");
    this.runtime.executeSimulationSteps = () => {};
    this.frame = 0;
    this.playing = false;
    this.last = performance.now();
    this.tick = this.tick.bind(this);
    requestAnimationFrame(this.tick);
  }
  async load(result) {
    this.pending = (this.pending || Promise.resolve())
      .catch(() => {})
      .then(() => this.loadFrames(result));
    return this.pending;
  }
  async loadFrames(result) {
    this.playing = false;
    await this.runtime.stop();
    this.engine.pause();
    this.kind = result.embodiment;
    if (this.kind !== "human")
      throw Error("This demo accepts human motion only.");
    this.frames = humanJoints(result.features);
    this.frame = 0;
    this.fps = 20;
    let bodies = "";
    for (let i = 0; i < 22; i++)
      bodies += `<body name="joint${i}" mocap="true" pos="0 0 1"><geom type="sphere" size="0.028" rgba="0.84 0.91 0.83 1"/></body>`;
    bones.forEach(
      (b, i) =>
        (bodies += `<body name="bone${i}" mocap="true" pos="0 0 1"><geom type="capsule" size="0.02 1" rgba="${b.color}"/></body>`),
    );
    const xml = `<mujoco model="MotionGlot kinematic playback"><option gravity="0 0 0"/><visual><headlight diffuse="0.8 0.8 0.8" ambient="0.4 0.4 0.4"/><rgba haze="0.09 0.17 0.19 1"/></visual><asset><texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".10 .18 .20" rgb2=".12 .21 .23"/><material name="floor" texture="grid" texrepeat="20 20" reflectance="0"/></asset><worldbody><light pos="0 0 8"/><geom type="plane" size="20 20 .1" material="floor"/>${bodies}</worldbody></mujoco>`;
    const archive = zipSync({ "scene.xml": strToU8(xml) });
    await this.engine.loadScene({ model: archive.buffer });
    await this.runtime.stop();
    this.engine.pause();
    this.engine.interaction.setMode("view");
    this.runtime.dynamicBodyIds = new Set(
      Array.from(this.runtime.mjModel.body_mocapid, (m, i) =>
        m >= 0 ? i : -1,
      ).filter((i) => i >= 0),
    );
    this.runtime.renderer.setClearColor(0x182d31);
    this.engine.camera.setTracking(false);
    const points = this.frames.flat();
    const low = [0, 1, 2].map((k) => Math.min(...points.map((p) => p[k]))),
      high = [0, 1, 2].map((k) => Math.max(...points.map((p) => p[k])));
    const center = low.map((v, k) => (v + high[k]) / 2),
      extent = Math.max(...high.map((v, k) => v - low[k]));
    this.engine.camera.set({
      lookat: center,
      distance: Math.max(3.8, extent * 1.8),
      azimuth: 135,
      elevation: -20,
    });
    this.view = this.engine.camera.get();
    this.apply(0);
    this.playing = true;
    this.last = performance.now();
  }
  apply(index) {
    if (!this.frames || !this.runtime.mjData) return;
    this.frame = Math.max(
      0,
      Math.min(this.frames.length - 1, Math.floor(index)),
    );
    const rt = this.runtime,
      data = rt.mjData,
      model = rt.mjModel,
      pose = this.frames[this.frame];
    pose.forEach((p, i) => {
      data.mocap_pos.set(p, i * 3);
      data.mocap_quat.set([1, 0, 0, 0], i * 4);
    });
    bones.forEach((bone, i) => {
      const a = pose[bone.a],
        b = pose[bone.b],
        d = b.map((x, k) => x - a[k]),
        length = Math.hypot(...d);
      const m = 22 + i;
      data.mocap_pos.set(
        a.map((x, k) => (x + b[k]) / 2),
        m * 3,
      );
      const z = d.map((x) => x / (length || 1));
      let q = [1 + z[2], -z[1], z[0], 0];
      let norm = Math.hypot(...q);
      if (norm < 1e-8) {
        q = [0, 1, 0, 0];
        norm = 1;
      }
      data.mocap_quat.set(
        q.map((x) => x / norm),
        m * 4,
      );
      // Each joint has one geom; each bone has one geom. Plane is geom 0.
      model.geom_size[(1 + 22 + i) * 3 + 1] = length / 2;
    });
    rt.mujoco.mj_forward(model, data);
    rt.updateCachedState();
    // MuJoCo geom sizes are immutable in Three's already-built meshes.
    bones.forEach((b, i) => {
      const bodyId = 23 + i,
        group = rt.bodies[bodyId];
      if (group) {
        const len = Math.hypot(...pose[b.a].map((x, k) => pose[b.b][k] - x));
        group.scale.set(1, len / 2, 1);
      }
    });
    this.onFrame?.(this.frame, this.frames.length, this.fps);
  }
  tick(now) {
    if (this.frames && this.playing) {
      const step = Math.floor(((now - this.last) * this.fps) / 1000);
      if (step) {
        this.apply((this.frame + step) % this.frames.length);
        this.last += (step * 1000) / this.fps;
      }
    } else this.last = now;
    requestAnimationFrame(this.tick);
  }
  resetCamera() {
    if (this.view) this.engine.camera.set(this.view);
  }
}
