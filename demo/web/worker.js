// Web worker: loads Python (Pyodide) + NumPy, the simulator code and the data,
// then runs experiments on request. Every round is posted back to the page.
const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.29.0/full/";
importScripts(PYODIDE_URL + "pyodide.js");

let py;
async function init() {
  postMessage({ type: "status", text: "Loading the Python runtime (first visit only, about 20 MB)…" });
  py = await loadPyodide({ indexURL: PYODIDE_URL });
  postMessage({ type: "status", text: "Loading NumPy…" });
  await py.loadPackage("numpy");
  postMessage({ type: "status", text: "Loading the simulator code and the UCI traffic data…" });
  const manifest = await (await fetch("py/manifest.json")).json();
  for (const f of manifest.files) {
    const buf = new Uint8Array(await (await fetch("py/" + f)).arrayBuffer());
    const parts = f.split("/");
    if (parts.length > 1) py.FS.mkdirTree("/home/pyodide/" + parts.slice(0, -1).join("/"));
    py.FS.writeFile("/home/pyodide/" + f, buf);
  }
  await py.runPythonAsync('import sys; sys.path.insert(0, "/home/pyodide"); import bridge; bridge.init()');
  postMessage({ type: "ready", info: JSON.parse(py.runPython("bridge.info_json()")),
                intro: py.runPython("bridge.intro_scene()") });
}
const ready = init().catch((e) => postMessage({ type: "error", text: String(e) }));

onmessage = async (ev) => {
  await ready;
  const m = ev.data;
  if (m.cmd !== "run") return;
  try {
    py.globals.set("_a", py.toPy(m));
    await py.runPythonAsync('bridge.run(_a["name"], int(_a["rounds"]), int(_a["seed"]), _a["title"], _a["job"])');
  } catch (e) {
    postMessage({ type: "error", text: String(e), job: m.job });
  }
};
