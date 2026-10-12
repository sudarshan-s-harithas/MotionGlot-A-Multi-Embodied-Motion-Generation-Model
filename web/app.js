import { MotionViewer } from "./viewer.js?v=human1";
import { parseNPY, validateFeatures } from "./motion.js?v=human1";
const $ = (id) => document.getElementById(id),
  viewer = new MotionViewer();
let worker,
  busy = false,
  current,
  examples = [],
  viewerReady = false;
const suggestions = {
  "text-to-human": [
    "A person walks forward",
    "A person sits down",
    "A person waves with one hand",
  ],
  "question-to-human": [
    "How does a person practice in a gymnastics class?",
    "How does a person improve leg strength?",
    "How does a person play golf?",
  ],
};
function status(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}
function updateTask(autoShow = true) {
  const task = $("task").value,
    caption = task === "human-to-text";
  $("text-input").hidden = caption;
  $("motion-input").hidden = !caption;
  $("suggestions").replaceChildren();
  for (const text of suggestions[task] || []) {
    const button = document.createElement("button");
    button.textContent = text;
    button.onclick = () => {
      $("prompt").value = text;
    };
    $("suggestions").append(button);
  }
  if (suggestions[task]) $("prompt").value = suggestions[task][0];
  if (caption && autoShow) {
    const embodiment = "human";
    if (current?.embodiment !== embodiment) {
      const example = examples.find((x) => x.embodiment === embodiment);
      if (example) show(example);
    }
  }
}
async function show(result) {
  current = result;
  try {
    if (viewerReady) await viewer.load(result);
    $("viewer-message").hidden = viewerReady;
    $("kind").textContent = "HUMAN MOTION";
    $("origin").textContent =
      result.origin === "generated"
        ? "GENERATED IN YOUR BROWSER"
        : "CHECKPOINT EXAMPLE";
    $("result-label").textContent =
      result.source || result.title || result.task;
    $("caption").hidden = !result.caption;
    $("caption").textContent = result.caption || "";
    $("download").disabled = false;
    $("timeline").max = result.features.length - 1;
    $("play").textContent = "Ⅱ";
  } catch (error) {
    $("viewer-message").hidden = false;
    $("viewer-message").textContent = "Unable to load the 3D viewer.";
    status(error.message, true);
  }
}
function setBusy(value) {
  busy = value;
  $("generate").disabled = value;
  $("cancel").hidden = !value;
  $("task").disabled = value;
  $("progress").hidden = !value;
  $("motion-file").disabled = value;
  for (const button of document.querySelectorAll(".example-card"))
    button.disabled = value;
}
function startWorker() {
  worker = new Worker(new URL("./worker.js?v=human1", import.meta.url), {
    type: "module",
  });
  worker.onmessage = async ({ data }) => {
    if (data.type === "status") {
      status(data.message);
      $("progress").hidden = data.progress === null;
      if (data.progress !== null) $("progress").value = data.progress;
    } else if (data.type === "error") {
      setBusy(false);
      status(data.message, true);
    } else if (data.type === "result") {
      await show(data.result);
      setBusy(false);
      $("generate").innerHTML = "Generate again <span>↗</span>";
      status(
        `${data.result.caption ? "Caption ready." : `${data.result.features.length} frames generated.`} ${data.result.terminated ? "End marker reached." : "Token limit reached; output may be incomplete."}`,
      );
    }
  };
  worker.onerror = (event) => {
    setBusy(false);
    status(
      event.message ||
        "The browser inference worker stopped. Try again or use the local Python demo.",
      true,
    );
  };
}
$("generate").onclick = async () => {
  if (busy) return;
  try {
    const task = $("task").value;
    let features;
    if (task.endsWith("-to-text")) {
      const file = $("motion-file").files[0];
      if (file) {
        if (file.size > 20e6)
          throw Error("Choose a motion file smaller than 20 MB.");
        features = file.name.endsWith(".npy")
          ? parseNPY(await file.arrayBuffer())
          : JSON.parse(await file.text());
        features = features.features || features;
      } else {
        const embodiment = "human";
        if (current?.embodiment !== embodiment)
          throw Error(
            "Select an example of the correct embodiment or upload a motion.",
          );
        features = current.features;
      }
      validateFeatures(features, 263);
    }
    if (task !== "human-to-text" && !$("prompt").value.trim())
      throw Error("Enter a prompt.");
    setBusy(true);
    status("Preparing browser inference…");
    if (!worker) startWorker();
    worker.postMessage({
      task,
      text: $("prompt").value,
      features,
      maxTokens: Number($("length").value),
      sample: $("sampling").checked,
      seed: Math.floor(Math.random() * 1e9),
    });
  } catch (error) {
    setBusy(false);
    status(error.message, true);
  }
};
$("cancel").onclick = () => {
  worker?.terminate();
  worker = null;
  setBusy(false);
  status(
    "Generation cancelled. Downloaded models remain cached when storage permits.",
  );
};
$("task").onchange = () => updateTask();
$("play").onclick = () => {
  viewer.playing = !viewer.playing;
  $("play").textContent = viewer.playing ? "Ⅱ" : "▶";
  $("play").setAttribute(
    "aria-label",
    viewer.playing ? "Pause playback" : "Play motion",
  );
};
$("timeline").oninput = () => {
  viewer.playing = false;
  $("play").textContent = "▶";
  viewer.apply(Number($("timeline").value));
};
$("camera").onclick = () => viewer.resetCamera();
viewer.onFrame = (frame, total, fps) => {
  $("timeline").max = total - 1;
  $("timeline").value = frame;
  $("time").textContent = (frame / fps).toFixed(2) + " s";
};
$("download").onclick = () => {
  if (!current) return;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(current)], { type: "application/json" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = `motionglot-${current.task || current.embodiment}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
try {
  examples = await (await fetch("./assets/examples.json?v=human1")).json();
  for (const example of examples) {
    const button = document.createElement("button");
    button.className = "example-card";
    const label = document.createElement("small");
    label.textContent = example.label;
    const title = document.createElement("strong");
    title.textContent = example.title;
    const arrow = document.createElement("span");
    arrow.textContent = "↗";
    button.append(label, title, arrow);
    button.onclick = async () => {
      $("task").value = $("task").value.endsWith("-to-text")
        ? `${example.embodiment}-to-text`
        : example.task;
      updateTask(false);
      $("prompt").value = example.source;
      await show(example);
      status("Showing a checkpoint example. Generate to try your own input.");
    };
    $("examples").append(button);
  }
  updateTask();
  await viewer.init($("viewer"));
  viewerReady = true;
  await show(examples[0]);
} catch (error) {
  $("viewer-message").textContent =
    "The viewer needs WebGL and access to the mjswan CDN.";
  status(error.message, true);
}
// Expose the viewer only for reproducible browser smoke checks in development.
if (location.hostname === "127.0.0.1" || location.hostname === "localhost")
  window.motionglotDebug = {
    viewer,
    get current() {
      return current;
    },
  };
