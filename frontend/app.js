const API_BASE = "http://localhost:8000";
const outEl = document.getElementById("out");
const recStatusEl = document.getElementById("recStatus");

function log(v) { outEl.textContent = typeof v === "string" ? v : JSON.stringify(v, null, 2); }

async function postForm(url, form) {
  const res = await fetch(url, { method: "POST", body: form });
  try { return await res.json(); }
  catch { return { status: res.status, text: await res.text() }; }
}

/* Upload & Extract */
document.getElementById("btnUploadExtract").addEventListener("click", async () => {
  const f = document.getElementById("uploadFile").files[0];
  const user = document.getElementById("uploadUser").value || undefined;
  if (!f) return log("Select a file first");
  const fd = new FormData();
  fd.append("file", f, f.name);
  if (user) fd.append("user_id", user);
  log("Uploading and extracting...");
  try { log(await postForm(`${API_BASE}/extract`, fd)); } catch (e) { log("Error: " + e); }
});

/* Server-side recording */
document.getElementById("btnServerStart").addEventListener("click", async () => {
  const name = document.getElementById("serverOutName").value || undefined;
  const fd = new FormData();
  if (name) fd.append("output_name", name);
  log("Starting server recording...");
  try { log(await postForm(`${API_BASE}/record/start`, fd)); } catch (e) { log("Error: " + e); }
});

document.getElementById("btnServerStop").addEventListener("click", async () => {
  log("Stopping server recording...");
  try {
    const res = await fetch(`${API_BASE}/record/stop`, { method: "POST" });
    log(await res.json());
  } catch (e) { log("Error: " + e); }
});

/* Test with uploaded file */
document.getElementById("btnTestUpload").addEventListener("click", async () => {
  const user = document.getElementById("testUser").value;
  if (!user) return log("Enter user_id");
  const f = document.getElementById("testFile").files[0];
  const fd = new FormData();
  fd.append("user_id", user);
  if (f) fd.append("file", f, f.name);
  log("Running test...");
  try { log(await postForm(`${API_BASE}/test`, fd)); } catch (e) { log("Error: " + e); }
});

/* Browser recorder (generic) */
let mediaRecorder = null;
let recordedChunks = [];

document.getElementById("btnStartBrowserRec").addEventListener("click", async () => {
  if (!navigator.mediaDevices?.getUserMedia) return log("Browser recording not supported");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => { if (e.data && e.data.size) recordedChunks.push(e.data); };
    mediaRecorder.onstop = () => stream.getTracks().forEach(t => t.stop());
    mediaRecorder.start();
    recStatusEl.textContent = "recording...";
    log("Browser recording started");
  } catch (e) { log("Mic permission denied or error: " + e); }
});

document.getElementById("btnStopBrowserRec").addEventListener("click", async () => {
  if (!mediaRecorder) return log("Not recording");
  mediaRecorder.stop();
  recStatusEl.textContent = "stopped";
  const blob = new Blob(recordedChunks, { type: 'audio/webm' });
  const filename = `browser_${Date.now()}.webm`;
  const fd = new FormData();
  fd.append("file", blob, filename);
  log("Uploading browser recording for extraction...");
  try { log(await postForm(`${API_BASE}/extract`, fd)); } catch (e) { log("Error: " + e); }
});

/* Test (record in browser then send to /test) */
document.getElementById("btnTestRecord").addEventListener("click", async () => {
  const user = document.getElementById("testUser").value;
  if (!user) return log("Enter user_id");
  if (!navigator.mediaDevices?.getUserMedia) return log("Browser recording not supported");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const rec = new MediaRecorder(stream);
    const chunks = [];
    rec.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
    rec.start();
    log("Recording 4 seconds for test...");
    await new Promise(r => setTimeout(r, 4000));
    rec.stop();
    await new Promise(resolve => rec.onstop = resolve);
    stream.getTracks().forEach(t => t.stop());
    const blob = new Blob(chunks, { type: 'audio/webm' });
    const fd = new FormData();
    fd.append("user_id", user);
    fd.append("file", blob, `test_${Date.now()}.webm`);
    log("Uploading test audio...");
    log(await postForm(`${API_BASE}/test`, fd));
  } catch (e) { log("Error: " + e); }
});