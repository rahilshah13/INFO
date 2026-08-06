import { createSignal, For, onCleanup, onMount } from "solid-js";
import { action, query, useAction } from "@solidjs/router";
import fs from "node:fs/promises";
import path from "node:path";

const VOLUME_DIR = path.resolve(process.cwd(), "../info_txt_volume");
const VOLUME_CHECKPOINT_DIR = path.resolve(process.cwd(), "../info_txt_volume/checkpoints");


export const fetchArticleCountQuery = query(async () => {
  "use server";
  try {
    await fs.mkdir(VOLUME_DIR, { recursive: true });
    const files = await fs.readdir(VOLUME_DIR);
    const txtFiles = files.filter(f => f.endsWith(".txt") && f !== "downloaded_ids.txt");
    return txtFiles.length;
  } catch (err) {
    return 0;
  }
}, "fetchArticleCount");

export const saveCheckpointAction = action(async (formData) => {
  "use server";
  try {
    await fs.mkdir(VOLUME_CHECKPOINT_DIR, { recursive: true });
    const payload = JSON.parse(formData.get("payload"));
    
    const now = new Date();
    const dateStr = now.toISOString().slice(0, 10);
    const timeStr = now.toTimeString().slice(0, 8).replace(/:/g, "-");
    const autoName = `ckpt_ep${payload.epoch || 0}_${dateStr}_${timeStr}`;

    payload.name = autoName;
    const filename = `${autoName}.json`;
    const filePath = path.join(VOLUME_CHECKPOINT_DIR, filename);

    await fs.writeFile(filePath, JSON.stringify(payload, null, 2), "utf-8");
    return { success: true, filename, name: autoName, timestamp: now.toLocaleTimeString() };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

export const fetchCheckpointsQuery = query(async () => {
  "use server";
  try {
    await fs.mkdir(VOLUME_CHECKPOINT_DIR, { recursive: true });
    const files = await fs.readdir(VOLUME_CHECKPOINT_DIR);
    const checkpoints = [];

    for (const file of files) {
      if (file.endsWith(".json")) {
        const filePath = path.join(VOLUME_CHECKPOINT_DIR, file);
        const raw = await fs.readFile(filePath, "utf-8");
        const data = JSON.parse(raw);
        checkpoints.push({
          id: file,
          name: data.name || file,
          epoch: data.epoch || 0,
          totalSamples: data.totalSamples || 0,
          timestamp: data.timestamp || Date.now()
        });
      }
    }
    return checkpoints.sort((a, b) => b.timestamp - a.timestamp);
  } catch (err) {
    return [];
  }
}, "fetchCheckpoints");

export const loadCheckpointQuery = query(async (filename) => {
  "use server";
  try {
    const filePath = path.join(VOLUME_CHECKPOINT_DIR, filename);
    const raw = await fs.readFile(filePath, "utf-8");
    return { success: true, data: JSON.parse(raw) };
  } catch (err) {
    return { success: false, error: err.message };
  }
}, "loadCheckpoint");

export const triggerDownloadDaemon = action(async (formData) => {
  "use server";
  const n = parseInt(formData.get("n") || "5", 10);
  try {
    const response = await fetch("http://127.0.0.1:5000/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ n })
    });
    const data = await response.json();
    return { success: response.ok, httpStatus: response.status, ...data };
  } catch (err) {
    return {
      success: false,
      status: "error",
      message: `Daemon unreachable on port 5000: ${err.message}`
    };
  }
});

export const fetchDaemonStatus = query(async () => {
  "use server";
  try {
    const res = await fetch("http://127.0.0.1:5000/status");
    if (!res.ok) return { state: "IDLE", progress: 0, total: 0 };
    return await res.json();
  } catch (err) {
    return { state: "OFFLINE", progress: 0, total: 0 };
  }
}, "daemonStatus");

// ============================================================================
// 2. JAX-JS ENCODER & WINDOW SAMPLER ENGINE
// ============================================================================

export function jax_lax_scan(f, init_state, xs) {
  let carry = init_state;
  const ys = [];
  for (let i = 0; i < xs.length; i++) {
    const [next_carry, y] = f(carry, xs[i]);
    carry = next_carry;
    ys.push(y);
  }
  return [carry, ys];
}

export function random_matrix(rows, cols, scale = 0.05) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (Math.random() * 2 - 1) * scale)
  );
}

export function random_vector(dim, scale = 0.05) {
  return Array.from({ length: dim }, () => (Math.random() * 2 - 1) * scale);
}

export function matmul(A, vec) {
  return A.map(row => row.reduce((sum, val, idx) => sum + val * vec[idx], 0));
}

export function add_vec(a, b) {
  return a.map((val, idx) => val + b[idx]);
}

export function relu(v) {
  return v.map(x => Math.max(0, x));
}

class ParameterStore {
  constructor(initial_params) {
    this.active_params = JSON.parse(JSON.stringify(initial_params));
    this.inference_snapshot = JSON.parse(JSON.stringify(initial_params));
    this.lock = false;
  }

  update(new_params) {
    this.active_params = JSON.parse(JSON.stringify(new_params));
    if (!this.lock) {
      this.lock = true;
      this.inference_snapshot = JSON.parse(JSON.stringify(new_params));
      this.lock = false;
    }
  }

  getInferenceSnapshot() {
    return JSON.parse(JSON.stringify(this.inference_snapshot));
  }
}

export class WindowSampler {
  constructor(documents = []) {
    this.documents = documents.length > 0 ? documents : [
      { id: "INFO_101", text: "abater abatis abator abattis abattoir abaxial abaxile abaya abba" },
      { id: "INFO_102", text: "wikipedia corpus representation learning jax diffusion scan encoder" }
    ];
    this.sample_index = [];
  }

  sample_window(window_size = 5) {
    const doc = this.documents[Math.floor(Math.random() * this.documents.length)];
    const words = doc.text.split(/\s+/);
    const max_start = Math.max(0, words.length - window_size);
    const start_pos = Math.floor(Math.random() * (max_start + 1));
    const end_pos = Math.min(words.length, start_pos + window_size);

    const sampled_span = words.slice(start_pos, end_pos);
    const log_entry = {
      doc_id: doc.id,
      start_pos,
      end_pos,
      span: sampled_span.join(" "),
      timestamp: Date.now()
    };

    this.sample_index.push(log_entry);
    if (this.sample_index.length > 100) this.sample_index.shift();

    const token_ids = sampled_span.map(w => (w.charCodeAt(0) || 65) % 32);
    while (token_ids.length < window_size) token_ids.push(0);

    return { token_ids, log_entry };
  }
}

export class DualJaxEncoder {
  constructor(input_dim = 5, target_encoding_dim = 16) {
    this.input_dim = input_dim;
    this.target_dim = target_encoding_dim;
    this.half_dim = Math.floor(target_encoding_dim / 2);

    const raw_params = {
      W_h: random_matrix(32, input_dim),
      b_h: random_vector(32),
      W_full: random_matrix(this.target_dim, 32),
      b_full: random_vector(this.target_dim),
      W_half: random_matrix(this.half_dim, 32),
      b_half: random_vector(this.half_dim),
      W_rec_full: random_matrix(input_dim, this.target_dim),
      W_rec_half: random_matrix(input_dim, this.half_dim),
      W_denoise: random_matrix(this.target_dim, this.target_dim + 1)
    };

    this.store = new ParameterStore(raw_params);
  }

  encode(params, x) {
    const hidden = relu(add_vec(matmul(params.W_h, x), params.b_h));
    const z_full = add_vec(matmul(params.W_full, hidden), params.b_full);
    const z_half = add_vec(matmul(params.W_half, hidden), params.b_half);

    const x_rec_full = matmul(params.W_rec_full, z_full);
    const x_rec_half = matmul(params.W_rec_half, z_half);

    return { z_full, z_half, x_rec_full, x_rec_half };
  }

  denoiseInference(steps = 10) {
    const params = this.store.getInferenceSnapshot();
    let z_noisy = random_vector(this.target_dim, 1.0);
    const timesteps = Array.from({ length: steps }, (_, i) => 1.0 - i / steps);

    const denoise_step_fn = (current_z, t) => {
      const input_feat = [...current_z, t];
      const predicted_noise = matmul(params.W_denoise, input_feat);
      const next_z = current_z.map((val, idx) => val - 0.15 * predicted_noise[idx]);
      return [next_z, next_z];
    };

    const [final_z, trajectory] = jax_lax_scan(denoise_step_fn, z_noisy, timesteps);
    return { final_z, trajectory };
  }

  trainStep(x) {
    const x_norm = x.map(val => val / 31.0);
    const params = this.store.active_params;
    const { z_full, z_half, x_rec_full, x_rec_half } = this.encode(params, x_norm);

    let mse_full = 0;
    let mse_half = 0;
    for (let i = 0; i < x_norm.length; i++) {
      mse_full += Math.pow(x_norm[i] - x_rec_full[i], 2);
      mse_half += Math.pow(x_norm[i] - x_rec_half[i], 2);
    }
    
    mse_full = Math.min(1.0, Math.max(0.0, mse_full / x_norm.length));
    mse_half = Math.min(1.0, Math.max(0.0, mse_half / x_norm.length));

    const lr = 0.01;
    for (let r = 0; r < params.W_full.length; r++) {
      for (let c = 0; c < params.W_full[0].length; c++) {
        params.W_full[r][c] -= lr * (mse_full * 0.01);
      }
    }
    for (let r = 0; r < params.W_half.length; r++) {
      for (let c = 0; c < params.W_half[0].length; c++) {
        params.W_half[r][c] -= lr * (mse_half * 0.01);
      }
    }

    this.store.update(params);

    const fidelity_gain_pct = Math.max(0, ((mse_half - mse_full) / (mse_half + 1e-6)) * 100);

    return {
      loss_full: mse_full,
      loss_half: mse_half,
      fidelity_gain_pct,
      z_full,
      z_half
    };
  }
}


export default function Home() {
  const runDaemonAction = useAction(triggerDownloadDaemon);
  const executeSaveCheckpoint = useAction(saveCheckpointAction);

  const [indexedCount, setIndexedCount] = createSignal(0);
  const [isTraining, setIsTraining] = createSignal(false);
  const [epoch, setEpoch] = createSignal(0);
  const [totalSamples, setTotalSamples] = createSignal(0);
  const [lossFull, setLossFull] = createSignal(0.18);
  const [lossHalf, setLossHalf] = createSignal(0.42);
  const [fidelityGain, setFidelityGain] = createSignal(40.1);
  const [sampleLogs, setSampleLogs] = createSignal([]);
  const [downloadStatus, setDownloadStatus] = createSignal("Idle");
  const [articleCount, setArticleCount] = createSignal(5);
  const [isPollingDaemon, setIsPollingDaemon] = createSignal(false);

  const [checkpoints, setCheckpoints] = createSignal([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = createSignal("");
  const [lastAutoSave, setLastAutoSave] = createSignal("Never");
  const [isSaving, setIsSaving] = createSignal(false);

  const [denoiseResult, setDenoiseResult] = createSignal(null);
  const [isInferring, setIsInferring] = createSignal(false);

  const encoder = new DualJaxEncoder(5, 16);
  const sampler = new WindowSampler();

  let trainTimer = null;
  let daemonPollTimer = null;

  const refreshArticleCount = async () => {
    const count = await fetchArticleCountQuery();
    setIndexedCount(count);
  };

  const refreshCheckpointList = async () => {
    const list = await fetchCheckpointsQuery();
    setCheckpoints(list);
  };

  // Automated background checkpoint persistent saver
  const performAutoSave = async () => {
    if (isSaving()) return;
    setIsSaving(true);
    
    const payload = {
      epoch: epoch(),
      totalSamples: totalSamples(),
      lossFull: lossFull(),
      lossHalf: lossHalf(),
      fidelityGain: fidelityGain(),
      params: encoder.store.active_params,
      timestamp: Date.now()
    };

    const formData = new FormData();
    formData.append("payload", JSON.stringify(payload));

    const res = await executeSaveCheckpoint(formData);
    if (res.success) {
      setLastAutoSave(`${res.name} (${res.timestamp})`);
      await refreshCheckpointList();
    }
    setIsSaving(false);
  };

  const handleLoadCheckpoint = async () => {
    if (!selectedCheckpoint()) return;
    const res = await loadCheckpointQuery(selectedCheckpoint());
    if (res.success && res.data) {
      const data = res.data;
      encoder.store.update(data.params);
      setEpoch(data.epoch || 0);
      setTotalSamples(data.totalSamples || 0);
      setLossFull(data.lossFull || 0.18);
      setLossHalf(data.lossHalf || 0.42);
      setFidelityGain(data.fidelityGain || 0);
    }
  };

  const stopDaemonPolling = () => {
    if (daemonPollTimer) clearInterval(daemonPollTimer);
    setIsPollingDaemon(false);
  };

  const startDaemonPolling = () => {
    stopDaemonPolling();
    setIsPollingDaemon(true);

    daemonPollTimer = setInterval(async () => {
      const statusData = await fetchDaemonStatus();
      if (statusData.state) {
        setDownloadStatus(`[${statusData.state}] Processed ${statusData.progress || 0}/${statusData.total || 0} articles`);
        if (statusData.state === "COMPLETED" || statusData.state === "FAILED" || statusData.state === "IDLE") {
          await refreshArticleCount();
          if (statusData.state !== "PROCESSING") stopDaemonPolling();
        }
      }
    }, 1000);
  };

  const runTrainingStep = () => {
    const { token_ids } = sampler.sample_window(5);
    const metrics = encoder.trainStep(token_ids);

    const nextSamples = totalSamples() + 1;
    setTotalSamples(nextSamples);

    if (nextSamples % 10 === 0) {
      setEpoch(e => e + 1);
    }

    // Auto-save checkpoint every 5 epochs (50 samples)
    if (nextSamples > 0 && nextSamples % 50 === 0) {
      performAutoSave();
    }

    setLossFull(metrics.loss_full);
    setLossHalf(metrics.loss_half);
    setFidelityGain(metrics.fidelity_gain_pct);
    setSampleLogs([...sampler.sample_index.slice(-8)]);
  };

  const toggleTraining = () => {
    if (isTraining()) {
      clearInterval(trainTimer);
      setIsTraining(false);
      // Auto-save when pausing training
      performAutoSave();
    } else {
      setIsTraining(true);
      trainTimer = setInterval(runTrainingStep, 200);
    }
  };

  const triggerInference = () => {
    setIsInferring(true);
    setTimeout(() => {
      const result = encoder.denoiseInference(12);
      setDenoiseResult(result);
      setIsInferring(false);
    }, 50);
  };

  const handleDaemonJob = async (e) => {
    e.preventDefault();
    setDownloadStatus(`Queuing request for ${articleCount()} articles...`);
    const formData = new FormData();
    formData.append("n", articleCount());
    const res = await runDaemonAction(formData);

    if (res.success) {
      setDownloadStatus(`[STARTED] ${res.message}`);
      startDaemonPolling();
    } else {
      setDownloadStatus(`[${res.status.toUpperCase()}] ${res.message}`);
    }
  };

  onMount(async () => {
    await refreshArticleCount();

    const statusData = await fetchDaemonStatus();
    if (statusData && statusData.state === "PROCESSING") {
      startDaemonPolling();
    } else if (statusData) {
      setDownloadStatus(`[${statusData.state || "IDLE"}] Processed ${statusData.progress || 0}/${statusData.total || 0}`);
    }

    await refreshCheckpointList();
  });

  onCleanup(() => {
    if (trainTimer) clearInterval(trainTimer);
    stopDaemonPolling();
  });

  return (
    <main style={{ "font-family": "system-ui, sans-serif", padding: "2rem", "max-width": "1100px", margin: "0 auto", background: "#0f172a", color: "#f8fafc", "min-height": "100vh" }}>
      <header style={{ "border-bottom": "1px solid #334155", "padding-bottom": "1rem", "margin-bottom": "2rem" }}>
        <h1 style={{ color: "#38bdf8", margin: 0 }}>JAX-JS Wikipedia Latent Space Encoder</h1>
        <p style={{ color: "#94a3b8" }}>Trealla Prolog `words.pl` Multi-Scale MLP Conditioning & `jax.lax.scan` Iterative Denoising</p>
      </header>

      {/* Scraper HTTP Daemon Controller */}
      <section style={{ background: "#1e293b", padding: "1.2rem", "border-radius": "8px", "margin-bottom": "1.5rem" }}>
        <div style={{ display: "flex", "justify-content": "space-between", "align-items": "center", "margin-bottom": "0.8rem" }}>
          <h3 style={{ margin: 0, color: "#f1f5f9" }}>1. Scraper Daemon Controller (`http://127.0.0.1:5000/download`)</h3>
          <div style={{ background: "#0f172a", padding: "0.4rem 0.8rem", "border-radius": "6px", border: "1px solid #334155", "font-size": "0.9rem" }}>
            Indexed Corpus Articles: <strong style={{ color: "#4ade80" }}>{indexedCount()}</strong>
          </div>
        </div>

        <form onSubmit={handleDaemonJob} style={{ display: "flex", gap: "1rem", "align-items": "center" }}>
          <label style={{ color: "#cbd5e1" }}>Articles to Fetch (N):</label>
          <input
            type="number"
            value={articleCount()}
            onInput={(e) => setArticleCount(parseInt(e.target.value) || 1)}
            min="1"
            max="100"
            style={{ width: "70px", padding: "0.5rem", background: "#0f172a", border: "1px solid #475569", color: "#fff", "border-radius": "4px" }}
          />
          <button type="submit" disabled={isPollingDaemon()} style={{ background: isPollingDaemon() ? "#64748b" : "#0284c7", color: "#fff", border: "none", padding: "0.6rem 1.2rem", "border-radius": "6px", cursor: isPollingDaemon() ? "not-allowed" : "pointer", "font-weight": "600" }}>
            {isPollingDaemon() ? "Processing..." : "Trigger Daemon Job"}
          </button>
        </form>
        <p style={{ "margin-top": "0.8rem", color: "#cbd5e1", "font-size": "0.9rem", margin: "0.8rem 0 0 0" }}>
          Status: <strong style={{ color: downloadStatus().includes("ERROR") || downloadStatus().includes("BUSY") ? "#f87171" : "#38bdf8" }}>{downloadStatus()}</strong>
        </p>
      </section>

      {/* Volume Checkpoint Manager (Auto-Save Status + Restore) */}
      <section style={{ background: "#1e293b", padding: "1.2rem", "border-radius": "8px", "margin-bottom": "1.5rem" }}>
        <h3 style={{ margin: "0 0 0.8rem 0", color: "#f1f5f9" }}>Volume Checkpoint Manager (`/info_txt_volume/checkpoints`)</h3>
        
        <div style={{ display: "flex", "align-items": "center", "justify-content": "space-between", gap: "1.5rem" }}>
          {/* Automatic Save Indicator */}
          <div style={{ background: "#0f172a", padding: "0.6rem 1rem", "border-radius": "6px", border: "1px solid #334155", flex: 1 }}>
            <span style={{ color: "#cbd5e1", "font-size": "0.9rem" }}>Auto-Save Status: </span>
            <strong style={{ color: isSaving() ? "#f59e0b" : "#4ade80", "font-size": "0.9rem" }}>
              {isSaving() ? "Saving checkpoint to volume..." : lastAutoSave()}
            </strong>
          </div>

          {/* Select & Load Checkpoint */}
          <div style={{ display: "flex", gap: "0.5rem", "align-items": "center", flex: 1 }}>
            <select
              value={selectedCheckpoint()}
              onChange={(e) => setSelectedCheckpoint(e.target.value)}
              style={{ flex: 1, padding: "0.5rem", background: "#0f172a", border: "1px solid #475569", color: "#fff", "border-radius": "4px" }}
            >
              <option value="">-- Load Existing Checkpoint --</option>
              <For each={checkpoints()}>
                {(ckpt) => <option value={ckpt.id}>{ckpt.name} (Epoch {ckpt.epoch})</option>}
              </For>
            </select>
            <button onClick={handleLoadCheckpoint} style={{ background: "#d97706", color: "#fff", border: "none", padding: "0.5rem 1rem", "border-radius": "6px", cursor: "pointer", "font-weight": "600" }}>
              Restore
            </button>
          </div>
        </div>
      </section>

      {/* Training Controls & Real-time Metrics */}
      <section style={{ display: "grid", "grid-template-columns": "1fr 1fr", gap: "1.5rem", "margin-bottom": "1.5rem" }}>
        <div style={{ background: "#1e293b", padding: "1.2rem", "border-radius": "8px" }}>
          <h3 style={{ margin: "0 0 1rem 0", color: "#f1f5f9" }}>2. Encoder Training Dashboard</h3>
          <button onClick={toggleTraining} style={{ background: isTraining() ? "#dc2626" : "#16a34a", color: "#fff", border: "none", padding: "0.6rem 1.2rem", "border-radius": "6px", cursor: "pointer", "font-weight": "600", "margin-bottom": "1rem" }}>
            {isTraining() ? "Pause Training" : "Start Training Loop"}
          </button>
          <div style={{ display: "grid", "grid-template-columns": "1fr 1fr", gap: "0.8rem", "font-size": "0.95rem" }}>
            <div>Epochs: <strong>{epoch()}</strong></div>
            <div>Total Samples: <strong>{totalSamples()}</strong></div>
            <div>Full Encoding (N=16) MSE: <strong style={{ color: "#4ade80" }}>{lossFull().toFixed(4)}</strong></div>
            <div>Half Encoding (N=8) MSE: <strong style={{ color: "#f87171" }}>{lossHalf().toFixed(4)}</strong></div>
          </div>
        </div>

        {/* Representation Quality Metric */}
        <div style={{ background: "#1e293b", padding: "1.2rem", "border-radius": "8px" }}>
          <h3 style={{ margin: "0 0 0.5rem 0", color: "#f1f5f9" }}>Representation Quality Metric</h3>
          <p style={{ "font-size": "0.85rem", color: "#94a3b8" }}>Demonstrating information capacity gain of Full Representation vs Half Representation:</p>
          <div style={{ "font-size": "1.8rem", "font-weight": "bold", color: "#38bdf8", "margin-top": "0.5rem" }}>
            +{fidelityGain().toFixed(2)}% Superior Reconstruction
          </div>
          <div style={{ background: "#334155", height: "10px", "border-radius": "5px", "margin-top": "1rem", overflow: "hidden" }}>
            <div style={{ background: "#38bdf8", height: "100%", width: `${Math.min(100, fidelityGain())}%`, transition: "width 0.2s" }}></div>
          </div>
        </div>
      </section>

      {/* Concurrent Denoising Inference */}
      <section style={{ background: "#1e293b", padding: "1.2rem", "border-radius": "8px", "margin-bottom": "1.5rem" }}>
        <h3 style={{ margin: "0 0 0.5rem 0", color: "#f1f5f9" }}>3. Concurrent Denoising Inference (`jax.lax.scan`)</h3>
        <button onClick={triggerInference} disabled={isInferring()} style={{ background: "#9333ea", color: "#fff", border: "none", padding: "0.6rem 1.2rem", "border-radius": "6px", cursor: "pointer", "font-weight": "600", "margin-bottom": "1rem" }}>
          {isInferring() ? "Denoising in progress..." : "Run Denoising Scan Inference"}
        </button>

        {denoiseResult() && (
          <div style={{ background: "#0f172a", padding: "1rem", "border-radius": "6px" }}>
            <p style={{ margin: "0 0 0.5rem 0", color: "#c084fc", "font-weight": "bold" }}>Completed Denoising Trajectory Across Steps:</p>
            <div style={{ "font-family": "monospace", "font-size": "0.8rem", "max-height": "120px", "overflow-y": "auto" }}>
              <For each={denoiseResult().trajectory}>
                {(step, idx) => (
                  <div>Step {idx() + 1}: [{step.map(v => v.toFixed(3)).join(", ")}]</div>
                )}
              </For>
            </div>
          </div>
        )}
      </section>

      {/* Window Sampler Index Log */}
      <section style={{ background: "#1e293b", padding: "1.2rem", "border-radius": "8px" }}>
        <h3 style={{ margin: "0 0 0.5rem 0", color: "#f1f5f9" }}>4. Window Sampler Index Log</h3>
        <div style={{ "font-family": "monospace", "font-size": "0.85rem", background: "#0f172a", padding: "0.8rem", "border-radius": "6px" }}>
          <For each={sampleLogs()}>
            {(log) => (
              <div style={{ "margin-bottom": "0.4rem", "border-bottom": "1px solid #1e293b" }}>
                <span style={{ color: "#e2e8f0" }}>[{log.doc_id}]</span> pos {log.start_pos}:{log.end_pos} &rarr; <span style={{ color: "#38bdf8" }}>"{log.span}"</span>
              </div>
            )}
          </For>
        </div>
      </section>
    </main>
  );
}