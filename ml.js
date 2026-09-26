import { numpy as np, jit, vmap } from "@jax-js/jax";
import { action, query } from "@solidjs/router";
import fs from "node:fs/promises";
import path from "node:path";

const VOLUME_DIR = path.resolve(process.cwd(), "../info_txt_volume");
const VOLUME_CHECKPOINT_DIR = path.resolve(process.cwd(), "../info_txt_volume/checkpoints");

export const fetchArticleCountQuery = query(async () => {
  "use server";
  return fs.mkdir(VOLUME_DIR, { recursive: true })
    .then(() => fs.readdir(VOLUME_DIR))
    .then(files => files.filter(f => f.endsWith(".txt") && f !== "downloaded_ids.txt").length)
    .catch(() => 0);
}, "fetchArticleCount");

export const fetchVocabQuery = query(async () => {
  "use server";
  const filePath = path.resolve(process.cwd(), "words.pl");
  const volPath = path.resolve(VOLUME_DIR, "words.pl");
  
  return fs.readFile(filePath, "utf-8")
    .catch(() => fs.readFile(volPath, "utf-8"))
    .then(raw => {
      const words = raw.split(/\r?\n/).map(w => w.trim()).filter(Boolean);
      return words.length > 0 ? words : null;
    })
    .catch(() => null);
}, "fetchVocab");

export const fetchArticleCorpusQuery = query(async () => {
  "use server";
  return fs.mkdir(VOLUME_DIR, { recursive: true })
    .then(() => fs.readdir(VOLUME_DIR))
    .then(files => files.filter(f => f.endsWith(".txt") && f !== "downloaded_ids.txt"))
    .then(txtFiles => Promise.all(txtFiles.map(file => fs.readFile(path.join(VOLUME_DIR, file), "utf-8").then(content => ({ filename: file, content })))))
    .catch(() => []);
}, "fetchArticleCorpus");

export const fetchPrologStatsQuery = query(async () => {
  "use server";
  return fetch("http://127.0.0.1:5000/metrics")
    .then(res => res.ok ? res.json() : Promise.reject())
    .then(metrics => ({
      totalEntries: metrics.vocabulary_size || 0,
      foundVocabCount: metrics.found_vocabulary_count || 0,
      coveragePct: metrics.coverage_percentage || 0,
      topTenWords: metrics.top_ten_words || [],
      foundPosCounts: metrics.found_pos_counts || {},
      totalPosCounts: metrics.total_pos_counts || {},
      sampleEntries: metrics.top_ten_words?.map(t => `${t.word} (${t.count})`) || []
    }))
    .catch(() => {
      const filePath = path.resolve(process.cwd(), "words.pl");
      const volPath = path.resolve(VOLUME_DIR, "words.pl");
      return fs.readFile(filePath, "utf-8")
        .catch(() => fs.readFile(volPath, "utf-8"))
        .then(raw => {
          const lines = raw.split(/\r?\n/).filter(l => l.trim().length > 0);
          return { totalEntries: lines.length, foundVocabCount: 0, coveragePct: 0, topTenWords: [], foundPosCounts: {}, totalPosCounts: {}, sampleEntries: lines.slice(0, 5) };
        })
        .catch(() => ({ totalEntries: 0, foundVocabCount: 0, coveragePct: 0, topTenWords: [], foundPosCounts: {}, totalPosCounts: {}, sampleEntries: [] }));
    });
}, "fetchPrologStats");

export const fetchPaginatedArticlesQuery = query(async (searchQuery = "", pageNum = 1) => {
  "use server";
  try {
    const res = await fetch(`http://127.0.0.1:5000/articles/search?q=${encodeURIComponent(searchQuery)}&page=${pageNum}&limit=5`);
    if (res.ok) return await res.json();
  } catch {}
  return { items: [], total: 0, page: 1, limit: 5, pages: 1 };
}, "fetchPaginatedArticles");

export const saveCheckpointAction = action(async (formData) => {
  "use server";
  try {
    await fs.mkdir(VOLUME_CHECKPOINT_DIR, { recursive: true });
    const payload = JSON.parse(formData.get("payload"));
    const now = new Date();
    const autoName = `ckpt_ep${payload.epoch || 0}_${now.toISOString().slice(0, 10)}_${now.toTimeString().slice(0, 8).replace(/:/g, "-")}`;
    payload.name = autoName;
    const filePath = path.join(VOLUME_CHECKPOINT_DIR, `${autoName}.json`);
    await fs.writeFile(filePath, JSON.stringify(payload, null, 2), "utf-8");

    const files = (await fs.readdir(VOLUME_CHECKPOINT_DIR)).filter(f => f.endsWith(".json"));
    if (files.length > 10) {
      const fileStats = await Promise.all(files.map(async f => ({ p: path.join(VOLUME_CHECKPOINT_DIR, f), mtime: (await fs.stat(path.join(VOLUME_CHECKPOINT_DIR, f))).mtime.getTime() })));
      fileStats.sort((a, b) => b.mtime - a.mtime);
      await Promise.all(fileStats.slice(10).map(old => fs.unlink(old.p)));
    }
    return { success: true, filename: `${autoName}.json`, name: autoName, timestamp: now.toLocaleTimeString() };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

export const fetchCheckpointsQuery = query(async () => {
  "use server";
  return fs.mkdir(VOLUME_CHECKPOINT_DIR, { recursive: true })
    .then(() => fs.readdir(VOLUME_CHECKPOINT_DIR))
    .then(files => Promise.all(files.filter(f => f.endsWith(".json")).map(async file => {
      const data = JSON.parse(await fs.readFile(path.join(VOLUME_CHECKPOINT_DIR, file), "utf-8"));
      return { id: file, name: data.name || file, epoch: data.epoch || 0, totalSamples: data.totalSamples || 0, timestamp: data.timestamp || Date.now() };
    })))
    .then(checkpoints => checkpoints.sort((a, b) => b.timestamp - a.timestamp))
    .catch(() => []);
}, "fetchCheckpoints");

export const loadCheckpointQuery = query(async (filename) => {
  "use server";
  return fs.readFile(path.join(VOLUME_CHECKPOINT_DIR, filename), "utf-8")
    .then(data => ({ success: true, data: JSON.parse(data) }))
    .catch(err => ({ success: false, error: err.message }));
}, "loadCheckpoint");

export const triggerDownloadDaemon = action(async (formData) => {
  "use server";
  return fetch("http://127.0.0.1:5000/download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ n: parseInt(formData.get("n") || "5", 10) })
  })
    .then(async response => ({ success: response.ok, httpStatus: response.status, ...(await response.json()) }))
    .catch(err => ({ success: false, status: "error", message: `Daemon unreachable: ${err.message}` }));
});

export const fetchDaemonStatus = query(async () => {
  "use server";
  return fetch("http://127.0.0.1:5000/status")
    .then(res => res.ok ? res.json() : { state: "IDLE", progress: 0, total: 0 })
    .catch(() => ({ state: "OFFLINE", progress: 0, total: 0 }));
}, "daemonStatus");

export const performAutoSave = (isSavingSignal, setIsSaving, epochSignal, totalSamplesSignal, lossFullSignal, lossHalfSignal, fidelityGainSignal, encoderInstance, setLastAutoSave, refreshCheckpointListFn, executeSaveCheckpointFn) => {
  if (isSavingSignal()) return Promise.resolve();
  setIsSaving(true);
  const payload = { 
    epoch: epochSignal(), 
    totalSamples: totalSamplesSignal(), 
    lossFull: lossFullSignal(), 
    lossHalf: lossHalfSignal(), 
    fidelityGain: fidelityGainSignal(), 
    params: encoderInstance.store.active_params, 
    timestamp: Date.now() 
  };
  const formData = new FormData();
  formData.append("payload", JSON.stringify(payload));
  return executeSaveCheckpointFn(formData).then(res => {
    if (res.success) { 
      setLastAutoSave(`${res.name} (${res.timestamp})`); 
      refreshCheckpointListFn(); 
    }
    setIsSaving(false);
  });
};

export const handleLoadCheckpoint = (selectedCheckpointSignal, loadCheckpointQueryFn, encoderInstance, setEpoch, setTotalSamples, setLossFull, setLossHalf, setFidelityGain) => {
  const target = selectedCheckpointSignal();
  if (!target) return;
  loadCheckpointQueryFn(target).then(res => {
    if (res.success && res.data) {
      encoderInstance.store.update(res.data.params);
      setEpoch(res.data.epoch || 0);
      setTotalSamples(res.data.totalSamples || 0);
      setLossFull(res.data.lossFull || 0.18);
      setLossHalf(res.data.lossHalf || 0.42);
      setFidelityGain(res.data.fidelityGain || 0);
    }
  });
};

export const stopDaemonPolling = (timerRef, setIsPollingDaemon) => { 
  if (timerRef) clearInterval(timerRef); 
  setIsPollingDaemon(false); 
};

export const startDaemonPolling = (timerRef, setIsPollingDaemon, fetchDaemonStatusFn, setDownloadStatus, refreshDataFn, stopPollingFn) => {
  stopPollingFn(timerRef, setIsPollingDaemon);
  setIsPollingDaemon(true);
  return setInterval(() => {
    fetchDaemonStatusFn().then(statusData => {
      if (statusData.state) {
        setDownloadStatus(`[${statusData.state}] Processed ${statusData.progress || 0}/${statusData.total || 0}`);
        if (["COMPLETED", "FAILED", "IDLE"].includes(statusData.state)) {
          refreshDataFn();
          if (statusData.state !== "PROCESSING") stopPollingFn(timerRef, setIsPollingDaemon);
        }
      }
    });
  }, 1000);
};

export const tokensToText = (tokenIds, vocab) => {
  const dict = vocab?.length ? vocab : ["transformer", "attention", "weights", "matrix", "vector"];
  return tokenIds.map(id => dict[Math.abs(Math.round(id)) % dict.length]).join(" ");
};

export const random_matrix = (rows, cols, scale = 0.02) => np.array(Array.from({ length: rows * cols }, () => (Math.random() * 2 - 1) * scale), { dtype: np.float32 }).reshape([rows, cols]);
export const random_vector = (dim, scale = 0.02) => np.array(Array.from({ length: dim }, () => (Math.random() * 2 - 1) * scale), { dtype: np.float32 });

export const layerNorm = (x, eps = 1e-5) => {
  const mean = x.ref.mean();
  const variance = x.ref.sub(mean.ref).square().mean();
  const normalized = x.ref.sub(mean.ref).div(variance.ref.add(eps).sqrt());
  mean.dispose();
  variance.dispose();
  return normalized;
};

export const softmax = (arr) => {
  const maxVal = arr.max().toArray()[0];
  const shifted = arr.ref.sub(maxVal);
  const exps = np.exp(shifted.ref);
  const sumExp = exps.ref.sum().toArray()[0] + 1e-9;
  return exps.div(sumExp);
};

export const multiHeadAttention = (Q, K, V, numHeads = 16) => {
  const dModel = Q.shape[0];
  const dHead = Math.floor(dModel / numHeads);
  const d_k = Math.sqrt(dHead) || 1;

  const headOutputs = Array.from({ length: numHeads }, (_, h) => {
    const start = h * dHead;
    const end = (h === numHeads - 1) ? dModel : (h + 1) * dHead;
    const Q_h = Q.ref.slice([start], [end]);
    const K_h = K.ref.slice([start], [end]);
    const V_h = V.ref.slice([start], [end]);

    const scores = Q_h.ref.matmul(K_h.ref.reshape([1, K_h.shape[0]])).div(d_k);
    const weights = softmax(scores);
    const headOut = V_h.ref.mul(weights.ref.toArray()[0] || 1.0);

    Q_h.dispose(); K_h.dispose(); V_h.dispose(); scores.dispose(); weights.dispose();
    return headOut;
  });

  const concatenated = np.concatenate(headOutputs);
  headOutputs.forEach(t => t.dispose());
  return concatenated;
};

class ParameterStore {
  constructor(initial_params) {
    this.active_params = initial_params;
    this.inference_snapshot = Object.fromEntries(Object.entries(initial_params).map(([k, v]) => [k, typeof v.copy === "function" ? v.copy() : v]));
    this.lock = false;
  }
  update(new_params) {
    this.active_params = new_params;
    if (!this.lock) {
      this.lock = true;
      Object.entries(new_params).forEach(([k, v]) => {
        if (this.inference_snapshot[k]?.dispose) this.inference_snapshot[k].dispose();
        this.inference_snapshot[k] = typeof v.copy === "function" ? v.copy() : v;
      });
      this.lock = false;
    }
  }
  getInferenceSnapshot() {
    return Object.fromEntries(Object.entries(this.inference_snapshot).map(([k, v]) => [k, typeof v.copy === "function" ? v.copy() : v]));
  }
}

export class WindowSampler {
  constructor(vocab = [], articles = []) {
    this.vocab = vocab;
    this.corpusItems = articles.length > 0 ? articles : [{ filename: "default.txt", content: "transformer attention weights matrix vector processing" }];
  }
  setCorpus(articles) { if (articles?.length) this.corpusItems = articles; }
  setVocab(vocab) { if (vocab?.length) this.vocab = vocab; }

  _getWords() {
    const item = this.corpusItems[Math.floor(Math.random() * this.corpusItems.length)];
    return {
      docId: item.filename || "article.txt",
      words: (item.content || "").replace(/[^\w\s]/gi, "").toLowerCase().split(/\s+/).filter(Boolean)
    };
  }

  sample_window() {
    const { docId, words } = this._getWords();
    const min_window = 40;
    const window_size = words.length >= min_window ? Math.floor(Math.random() * (words.length - min_window + 1)) + min_window : words.length;
    const start_pos = words.length >= min_window ? Math.floor(Math.random() * (words.length - window_size + 1)) : 0;
    const sampled_span = words.slice(start_pos, start_pos + window_size);

    while (sampled_span.length < min_window) sampled_span.push(this.vocab[Math.floor(Math.random() * this.vocab.length)] || "prediction");

    const span = sampled_span.join(" ");
    const token_ids = sampled_span.map(w => Math.abs([...w].reduce((hash, c) => (hash << 5) - hash + c.charCodeAt(0), 0)) % Math.max(1, this.vocab.length));
    return { token_ids, span, charLength: span.length, docId };
  }

  sample_inference_seed() {
    const { docId, words } = this._getWords();
    const window_size = 5;
    const start_pos = words.length > window_size ? Math.floor(Math.random() * (words.length - window_size)) : 0;
    const sampled_span = words.slice(start_pos, start_pos + window_size);

    while (sampled_span.length < window_size) sampled_span.push(this.vocab[Math.floor(Math.random() * this.vocab.length)] || "prediction");

    const span = sampled_span.join(" ");
    const token_ids = sampled_span.map(w => Math.abs([...w].reduce((hash, c) => (hash << 5) - hash + c.charCodeAt(0), 0)) % Math.max(1, this.vocab.length));
    return { token_ids, span, charLength: span.length, docId };
  }
}

export class TransformerDiffusionEncoder {
  constructor(input_dim = 1000, target_encoding_dim = 1000, vocab = []) {
    this.target_dim = target_encoding_dim;
    this.half_dim = Math.floor(target_encoding_dim / 2);
    this.vocab = vocab.length > 0 ? vocab : ["transformer", "attention", "weights", "matrix", "vector"];

    this.store = new ParameterStore({
      W_q: random_matrix(this.target_dim, this.target_dim, 0.01),
      W_k: random_matrix(this.target_dim, this.target_dim, 0.01),
      W_v: random_matrix(this.target_dim, this.target_dim, 0.01),
      W_h: random_matrix(256, this.target_dim, 0.01),
      b_h: random_vector(256, 0.01),
      W_full: random_matrix(this.target_dim, 256, 0.01),
      b_full: random_vector(this.target_dim, 0.01),
      W_half: random_matrix(this.half_dim, 256, 0.01),
      b_half: random_vector(this.half_dim, 0.01),
      W_rec_full: random_matrix(this.target_dim, this.target_dim, 0.01),
      W_rec_half: random_matrix(this.target_dim, this.half_dim, 0.01),
      W_denoise: random_matrix(this.target_dim, this.target_dim + 1, 0.01)
    });

    this._jitEncode = jit((params, x_padded) => {
      const Q = params.W_q.ref.matmul(x_padded.ref);
      const K = params.W_k.ref.matmul(x_padded.ref);
      const V = params.W_v.ref.matmul(x_padded.ref);
      
      const attended = layerNorm(multiHeadAttention(Q, K, V, 16));
      const hidden = np.maximum(0, layerNorm(params.W_h.ref.matmul(attended).add(params.b_h.ref)));
      
      const z_full = layerNorm(params.W_full.ref.matmul(hidden.ref).add(params.b_full.ref));
      const z_half = layerNorm(params.W_half.ref.matmul(hidden.ref).add(params.b_half.ref));
      
      const x_rec_full = params.W_rec_full.ref.matmul(z_full.ref);
      const x_rec_half = params.W_rec_half.ref.matmul(z_half.ref);

      Q.dispose(); K.dispose(); V.dispose(); attended.dispose(); hidden.dispose();
      return { z_full, z_half, x_rec_full, x_rec_half };
    });
  }

  setVocab(vocab) { if (vocab?.length) this.vocab = vocab; }
  encode(params, x_padded) { return this._jitEncode(params, x_padded); }

  diffusionInversionAndSample(steps = 32, samplerInstance = null, targetCompletionTokens = 16, temperature = 0.8, topK = 50, topP = 0.9, seedTokenIds = null) {
    const params = this.store.getInferenceSnapshot();
    const sampler = samplerInstance || new WindowSampler(this.vocab);
    
    const sampled = seedTokenIds?.length ? { token_ids: seedTokenIds, docId: "custom_seed" } : sampler.sample_inference_seed();
    const seedDecoded = tokensToText(sampled.token_ids, this.vocab);
    
    let current_z_arr = Array.from({ length: this.target_dim }, (_, idx) => idx < sampled.token_ids.length ? (sampled.token_ids[idx] / Math.max(1, this.vocab.length)) * 2.0 - 1.0 : 0);
    const mockPartsOfSpeech = ["n", "v", "adj", "adv", "pron", "det", "prep", "conj"];

    const sampleNextTokenId = (logitsArr) => {
      let indexedProbs = softmax(np.array(logitsArr.map(l => l / Math.max(1e-5, temperature)), { dtype: np.float32 })).toArray().map((p, idx) => ({ p, idx }));
      
      if (topK > 0 && topK < indexedProbs.length) {
        indexedProbs.sort((a, b) => b.p - a.p);
        indexedProbs = indexedProbs.slice(0, topK);
        const sumP = indexedProbs.reduce((acc, item) => acc + item.p, 1e-9);
        indexedProbs.forEach(item => item.p /= sumP);
      } else if (topP > 0 && topP < 1.0) {
        indexedProbs.sort((a, b) => b.p - a.p);
        let cumSum = 0, cutIndex = indexedProbs.length;
        for (let i = 0; i < indexedProbs.length; i++) {
          cumSum += indexedProbs[i].p;
          if (cumSum > topP) { cutIndex = i + 1; break; }
        }
        indexedProbs = indexedProbs.slice(0, cutIndex);
        const sumP = indexedProbs.reduce((acc, item) => acc + item.p, 1e-9);
        indexedProbs.forEach(item => item.p /= sumP);
      }
      
      const r = Math.random();
      let cumulative = 0;
      for (const item of indexedProbs) {
        cumulative += item.p;
        if (r <= cumulative) return item.idx % Math.max(1, this.vocab.length);
      }
      return indexedProbs[0]?.idx % Math.max(1, this.vocab.length) || 0;
    };

    const trajectoryFull = Array.from({ length: steps }, (_, i) => {
      const step = i + 1;
      const tVal = 1.0 - (i / (steps - 1 || 1));
      const inputVec = np.array([...current_z_arr, tVal], { dtype: np.float32 });
      const rawLogits = params.W_denoise.ref.matmul(inputVec).toArray();
      inputVec.dispose();
      
      current_z_arr = current_z_arr.map((val, idx) => (1.0 - tVal * 0.03) * val + 0.05 * rawLogits[idx % rawLogits.length]);
      const truncatedTokenIds = Array.from({ length: targetCompletionTokens }, (_, j) => sampleNextTokenId(rawLogits.map((l, lIdx) => l + Math.sin(j + lIdx + tVal))));

      return {
        step, t: Number(tVal.toFixed(2)), logits: rawLogits,
        pureCompletion: tokensToText(truncatedTokenIds, this.vocab),
        lexicalEntries: truncatedTokenIds.map((id, index) => [this.vocab[id % this.vocab.length] || "term", mockPartsOfSpeech[(id + index) % mockPartsOfSpeech.length]])
      };
    });

    Object.values(params).forEach(v => v.dispose());
    return { seedDecoded, docId: sampled.docId, trajectoryFull, trajectoryHalf: trajectoryFull };
  }

  trainStep(token_ids) {
    if (this._previousLoss === undefined) this._previousLoss = null;

    const x_padded_arr = Array.from({ length: this.target_dim }, (_, idx) => idx < token_ids.length ? (token_ids[idx] / this.vocab.length) * 2.0 - 1.0 : 0);
    const x_padded = np.array(x_padded_arr, { dtype: np.float32 });

    const params = this.store.active_params;
    const { z_full, z_half, x_rec_full, x_rec_half } = this.encode(params, x_padded);

    const xRecFullArr = x_rec_full.toArray(), xRecHalfArr = x_rec_half.toArray();
    let mse_full = 0, mse_half = 0;
    for (let i = 0; i < x_padded_arr.length; i++) {
      mse_full += Math.pow(x_padded_arr[i] - xRecFullArr[i], 2);
      mse_half += Math.pow(x_padded_arr[i] - xRecHalfArr[i], 2);
    }
    mse_full = Math.min(1, Math.max(0, mse_full / x_padded_arr.length));
    mse_half = Math.min(1, Math.max(0, mse_half / x_padded_arr.length));

    const lossDelta = this._previousLoss !== null ? this._previousLoss - mse_full : 0;
    this._previousLoss = mse_full;

    const diff = x_rec_full.ref.sub(x_padded);
    const grad = diff.reshape([diff.shape[0], 1]).matmul(z_full.ref.reshape([1, z_full.shape[0]]));
    this.store.update({ ...params, W_full: params.W_full.ref.sub(grad.mul(0.001)) });

    const decoded = tokensToText(z_full.toArray().map(v => Math.abs(Math.round((v + 1.0) * 0.5 * this.vocab.length))), this.vocab);

    [x_padded, z_full, z_half, x_rec_full, x_rec_half, diff, grad].forEach(t => t.dispose());

    return {
      loss_full: mse_full, loss_half: mse_half, loss_delta: lossDelta,
      is_converging: lossDelta >= 0,
      fidelity_gain_pct: Math.max(0, ((mse_half - mse_full) / (mse_half + 1e-6)) * 100),
      decoded
    };
  }
}