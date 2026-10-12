import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { MotionTokenizer } from "./tokenizer.js";
import {
  humanJoints,
  robotFrames,
  parseNPY,
  validateFeatures,
} from "./motion.js";
const fixtures = JSON.parse(
  await readFile(
    new URL("./assets/test-fixtures.json", import.meta.url),
    "utf8",
  ),
);
test("human skeleton agrees with Python HumanML3D recovery", () => {
  const actual = humanJoints(fixtures.features);
  for (let i = 0; i < actual.length; i++)
    for (let j = 0; j < 22; j++)
      for (let k = 0; k < 3; k++)
        assert.ok(Math.abs(actual[i][j][k] - fixtures.joints[i][j][k]) < 2e-5);
});
test("robot increments integrate x, y and heading", () => {
  assert.deepEqual(
    robotFrames([
      [1, 2, 0.1],
      [2, 1, 0.2],
    ]),
    [
      [1, 2, 0.1],
      [3, 3, 0.1 + 0.2],
    ],
  );
});
test("caption input rejects invalid shapes and nonfinite values", () => {
  assert.throws(() => validateFeatures([[1, 2, 3]], 3));
  assert.throws(() =>
    validateFeatures(
      Array.from({ length: 4 }, () => [1, NaN, 3]),
      3,
    ),
  );
  assert.doesNotThrow(() =>
    validateFeatures(
      Array.from({ length: 4 }, () => [1, 2, 3]),
      3,
    ),
  );
  assert.throws(() => parseNPY(new ArrayBuffer(12)));
});
test("byte BPE agrees with the released Python tokenizer for all task markers and Unicode", async () => {
  const settings = JSON.parse(
    await readFile(new URL("./settings.json", import.meta.url), "utf8"),
  );
  const base = process.env.MOTIONGLOT_TEST_MODEL_BASE || settings.modelBase;
  const [config, json] = await Promise.all(
    ["browser-config.json", "tokenizer.json"].map(async (f) => {
      const r = await fetch(base + f);
      assert.ok(r.ok);
      return r.json();
    }),
  );
  const tokenizer = new MotionTokenizer(json, config.tokens);
  for (const row of fixtures.tokenization) {
    assert.deepEqual(tokenizer.encode(row.text), row.ids, row.text);
    assert.equal(tokenizer.decode(row.ids), row.decoded, row.text);
  }
});
