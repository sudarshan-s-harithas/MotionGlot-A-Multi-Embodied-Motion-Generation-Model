import { createEngine } from "https://cdn.jsdelivr.net/npm/mjswan@0.11.6/dist/mjswan.js";
import {
  zipSync,
  strToU8,
} from "https://cdn.jsdelivr.net/npm/fflate@0.8.2/+esm";
import { bones, humanJoints, robotFrames } from "./motion.js";

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
    this.frames =
      this.kind === "human"
        ? humanJoints(result.features)
        : robotFrames(result.features);
    this.frame = 0;
    this.fps = 20;
    let bodies = "";
    if (this.kind === "human") {
      for (let i = 0; i < 22; i++)
        bodies += `<body name="joint${i}" mocap="true" pos="0 0 1"><geom type="sphere" size="0.028" rgba="0.84 0.91 0.83 1"/></body>`;
      bones.forEach(
        (b, i) =>
          (bodies += `<body name="bone${i}" mocap="true" pos="0 0 1"><geom type="capsule" size="0.02 1" rgba="${b.color}"/></body>`),
      );
    } else {
      bodies =
        '<body name="robot" mocap="true" pos="0 0 0.45"><geom type="box" size="0.32 0.15 0.10" rgba="0.83 0.92 0.40 1"/><geom type="box" pos="0.28 0 0.02" size="0.10 0.11 0.07" rgba="0.23 0.34 0.36 1"/>';
      for (const x of [-0.23, 0.23])
        for (const y of [-0.15, 0.15])
          bodies += `<geom type="capsule" fromto="${x} ${y} 0 ${x - 0.05} ${y} -0.32" size="0.035" rgba="0.36 0.48 0.48 1"/>`;
      bodies += "</body>";
      // The trajectory consists of fixed cylinders, separate from the moving marker.
      const stride = Math.max(1, Math.ceil(this.frames.length / 160));
      for (let i = stride; i < this.frames.length; i += stride) {
        const a = this.frames[i - stride],
          b = this.frames[i];
        if (Math.hypot(a[0] - b[0], a[1] - b[1]) > 0.0001)
          bodies += `<geom type="capsule" fromto="${a[0]} ${a[1]} .018 ${b[0]} ${b[1]} .018" size=".018" rgba="0.42 0.69 0.71 1"/>`;
      }
      if (result.goal)
        bodies += `<geom type="box" pos="${-6.75 + Math.floor((result.goal[0] + 7) / 0.5) * 0.5} ${-6.75 + Math.floor((result.goal[1] + 7) / 0.5) * 0.5} .008" size=".25 .25 .008" rgba="0.84 0.91 0.36 0.6"/>`;
    }
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
    const points =
      this.kind === "human"
        ? this.frames.flat()
        : this.frames.map((p) => [p[0], p[1], 0.45]);
    const low = [0, 1, 2].map((k) => Math.min(...points.map((p) => p[k]))),
      high = [0, 1, 2].map((k) => Math.max(...points.map((p) => p[k])));
    const center = low.map((v, k) => (v + high[k]) / 2),
      extent = Math.max(...high.map((v, k) => v - low[k]));
    this.engine.camera.set({
      lookat: center,
      distance: Math.max(this.kind === "human" ? 3.8 : 4, extent * 1.8),
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
    if (this.kind === "human") {
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
    } else {
      data.mocap_pos.set([pose[0], pose[1], 0.45]);
      data.mocap_quat.set([Math.cos(pose[2] / 2), 0, 0, Math.sin(pose[2] / 2)]);
    }
    rt.mujoco.mj_forward(model, data);
    rt.updateCachedState();
    // MuJoCo geom sizes are immutable in Three's already-built meshes.
    if (this.kind === "human")
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
