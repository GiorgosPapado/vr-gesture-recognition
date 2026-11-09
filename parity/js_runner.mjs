// Runs the preprocessing in webxr/gesture_engine.js under Node, unchanged.
// The engine imports "../build/three.module.js" (absent in the repo). We write a copy of the
// engine to a temp dir with ONLY that import line rewritten to the npm `three` module, stub the
// `ort` global to capture the input tensor, and print it as JSON. No model is run.
// Usage: node parity/js_runner.mjs parity/fixtures/sample.json
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const enginePath = path.resolve(here, "..", "webxr", "gesture_engine.js");
const threePath = path.resolve(here, "node_modules", "three", "build", "three.module.js");
if (!fs.existsSync(threePath)) {
  console.error("three not installed: run `npm ci` in parity/");
  process.exit(2);
}

const original = "../build/three.module.js";
let src = fs.readFileSync(enginePath, "utf8");
if (!src.includes(original)) {
  console.error("engine import line changed; update js_runner.mjs");
  process.exit(2);
}
src = src.replace(original, pathToFileURL(threePath).href);
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "vrg-parity-"));
process.on("exit", () => fs.rmSync(tmp, { recursive: true, force: true }));
const tmpEngine = path.join(tmp, "gesture_engine.mjs");
fs.writeFileSync(tmpEngine, src);

let captured = null;
globalThis.ort = {
  Tensor: class {
    constructor(data, dims) {
      this.data = data;
      this.dims = dims;
      captured = { data, dims };
    }
  },
  InferenceSession: {
    create: async () => ({ run: async () => ({ output: { data: [1] } }) }),
  },
};

const THREE = await import(pathToFileURL(threePath).href);
const { GestureRecognizer } = await import(pathToFileURL(tmpEngine).href);

const fixture = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const [xs, ys, zs] = fixture.positions_xyz_by_time;
const results = {};
for (const c of fixture.cases) {
  const [ox, oy, oz] = c.position_offset_xyz || [0, 0, 0]; // optional constant offset added to ALL positions
  const positions = xs.map((x, i) => new THREE.Vector3(x + ox, ys[i] + oy, zs[i] + oz));
  const q = new THREE.Quaternion(...c.ref_orientation_xyzw);
  const orientations = positions.map(() => q.clone());
  await new GestureRecognizer().inference(positions, orientations, fixture.timestamps_synthetic);
  results[c.name] = { dims: Array.from(captured.dims), data: Array.from(captured.data) };
}
console.log(JSON.stringify(results));
