export const chains = [
  [0, 2, 5, 8, 11],
  [0, 1, 4, 7, 10],
  [0, 3, 6, 9, 12, 15],
  [9, 14, 17, 19, 21],
  [9, 13, 16, 18, 20],
];
export const bones = chains.flatMap((chain, c) =>
  chain.slice(1).map((j, i) => ({
    a: chain[i],
    b: j,
    color:
      c === 0 || c === 3
        ? "0.82 0.38 0.30 1"
        : c === 1 || c === 4
          ? "0.40 0.66 0.80 1"
          : "0.80 0.93 0.46 1",
  })),
);
export function humanJoints(rows) {
  let angle = 0,
    x = 0,
    z = 0;
  const frames = [];
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    if (i) angle += rows[i - 1][0];
    const c = Math.cos(2 * angle),
      s = Math.sin(2 * angle);
    if (i) {
      const p = rows[i - 1];
      x += c * p[1] - s * p[2];
      z += s * p[1] + c * p[2];
    }
    const joints = [[x, r[3], z]];
    for (let j = 0; j < 21; j++) {
      const k = 4 + j * 3,
        a = r[k],
        b = r[k + 1],
        d = r[k + 2];
      joints.push([c * a - s * d + x, b, s * a + c * d + z]);
    }
    frames.push(joints);
  }
  const floor = Math.min(...frames.map((j) => Math.min(...j.map((p) => p[1]))));
  return frames.map((js) => js.map((p) => [p[0], p[2], p[1] - floor]));
}
export function validateFeatures(rows, dim) {
  if (
    !Array.isArray(rows) ||
    rows.length < 4 ||
    rows.length > 3840 ||
    rows.length % 4 ||
    !rows.every(
      (r) => Array.isArray(r) && r.length === dim && r.every(Number.isFinite),
    )
  )
    throw Error(
      `Use finite [frames, ${dim}] features with 4–3840 frames, divisible by 4.`,
    );
  return rows;
}
export function parseNPY(buffer) {
  const a = new Uint8Array(buffer),
    view = new DataView(buffer);
  if (String.fromCharCode(...a.slice(1, 6)) !== "NUMPY")
    throw Error("Invalid NPY file.");
  const version = a[6],
    offset = version === 1 ? 10 : 12,
    len = version === 1 ? view.getUint16(8, true) : view.getUint32(8, true),
    header = new TextDecoder().decode(a.slice(offset, offset + len));
  if (!/['"]fortran_order['"]:\s*False/.test(header))
    throw Error("Use a C-order array.");
  const shape = header.match(/['"]shape['"]:\s*\((\d+),\s*(\d+),?\s*\)/);
  const dtype = header.match(/['"]descr['"]:\s*['"]([<>=|])([fi])(\d+)['"]/);
  if (!shape || !dtype || dtype[2] !== "f" || ![4, 8].includes(+dtype[3]))
    throw Error("Use a float32 or float64 [frames, features] array.");
  const [n, d] = shape.slice(1).map(Number),
    size = +dtype[3],
    start = offset + len;
  if (n * d * size !== buffer.byteLength - start)
    throw Error("NPY dimensions do not match data size.");
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: d }, (_, j) =>
      size === 4
        ? view.getFloat32(start + (i * d + j) * size, dtype[1] !== ">")
        : view.getFloat64(start + (i * d + j) * size, dtype[1] !== ">"),
    ),
  );
}
