import { createSignal, For, onCleanup, onMount } from "solid-js";
import { action, query, useAction } from "@solidjs/router";
import { numpy as np, jit, vmap, lax } from "@jax-js/jax";
import fs from "node:fs/promises";
import path from "node:path";

const VOLUME_DIR = path.resolve(process.cwd(), "../info_txt_volume");
const VOLUME_CHECKPOINT_DIR = path.resolve(process.cwd(), "../info_txt_volume/checkpoints");

export const fetchArticleCountQuery = query(async () => {
  "use server";
  try {
    await fs.mkdir(VOLUME_DIR, { recursive: true });
    const files = await fs.readdir(VOLUME_DIR);
    return files.filter(f => f.endsWith(".txt") && f !== "downloaded_ids.txt" && f !== "parsed_ids.txt").length;
  } catch {
    return 0;
  }
}, "fetchArticleCount");

export const fetchVocabQuery = query(async () => {
  "use server";
  try {
    let filePath = path.resolve(process.cwd(), "words.pl");
    let raw;
    try {
      raw = await fs.readFile(filePath, "utf-8");
    } catch {
      filePath = path.resolve(VOLUME_DIR, "words.pl");
      raw = await fs.readFile(filePath, "utf-8");
    }
    const words = raw.split(/\r?\n/).map(w => w.trim()).filter(w => w.length > 0);
    return words.length > 0 ? words : null;
  } catch {
    return null;
  }
}, "fetchVocab");

export const fetchArticleCorpusQuery = query(async () => {
  "use server";
  try {
    await fs.mkdir(VOLUME_DIR, { recursive: true });
    const files = await fs.readdir(VOLUME_DIR);
    const txtFiles = files.filter(f => f.endsWith(".txt") && f !== "downloaded_ids.txt" && f !== "parsed_ids.txt");
    const contents = [];
    for (const file of txtFiles) {
      const raw = await fs.readFile(path.join(VOLUME_DIR, file), "utf-8");
      contents.push({ filename: file, content: raw });
    }
    return contents;
  } catch {
    return [];
  }
}, "fetchArticleCorpus");

export const fetchPrologStatsQuery = query(async () => {
  "use server";
  try {
    const res = await fetch("http://127.0.0.1:5000/metrics");
    if (res.ok) {
      const metrics = await res.json();
      return {
        totalEntries: metrics.vocabulary_size || 0,
        foundVocabCount: metrics.found_vocabulary_count || 0,
        coveragePct: metrics.coverage_percentage || 0,
        topTenWords: metrics.top_ten_words || [],
        foundPosCounts: metrics.found_pos_counts || {},
        totalPosCounts: metrics.total_pos_counts || {},
        topTenSentenceFormats: metrics.top_ten_sentence_formats || [],
        topTenFlatPos: metrics.top_ten_flat_pos || [],
        avgSentenceLength: metrics.average_sentence_length || 0.0,
        uniqueSentenceFormatsCount: metrics.unique_sentence_formats_count || 0,
        sampleEntries: metrics.top_ten_words?.map(t => `${t.word} (${t.count})`) || [],
        status: "active vocabulary"
      };
    }
  } catch {}
  
  try {
    let filePath = path.resolve(process.cwd(), "words.pl");
    let raw;
    try {
      raw = await fs.readFile(filePath, "utf-8");
    } catch {
      filePath = path.resolve(VOLUME_DIR, "words.pl");
      raw = await fs.readFile(filePath, "utf-8");
    }
    const lines = raw.split(/\r?\n/).filter(l => l.trim().length > 0);
    return { totalEntries: lines.length, foundVocabCount: 0, coveragePct: 0, topTenWords: [], foundPosCounts: {}, totalPosCounts: {}, topTenSentenceFormats: [], topTenFlatPos: [], avgSentenceLength: 0.0, uniqueSentenceFormatsCount: 0, sampleEntries: lines.slice(0, 5), status: "active vocabulary" };
  } catch {
    return { totalEntries: 0, foundVocabCount: 0, coveragePct: 0, topTenWords: [], foundPosCounts: {}, totalPosCounts: {}, topTenSentenceFormats: [], topTenFlatPos: [], avgSentenceLength: 0.0, uniqueSentenceFormatsCount: 0, sampleEntries: [], status: "offline" };
  }
}, "fetchPrologStats");

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
      const fileStats = await Promise.all(files.map(async f => ({ f, p: path.join(VOLUME_CHECKPOINT_DIR, f), mtime: (await fs.stat(path.join(VOLUME_CHECKPOINT_DIR, f))).mtime.getTime() })));
      fileStats.sort((a, b) => b.mtime - a.mtime);
      for (const old of fileStats.slice(10)) await fs.unlink(old.p);
    }
    return { success: true, filename: `${autoName}.json`, name: autoName, timestamp: now.toLocaleTimeString() };
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
        const data = JSON.parse(await fs.readFile(path.join(VOLUME_CHECKPOINT_DIR, file), "utf-8"));
        checkpoints.push({ id: file, name: data.name || file, epoch: data.epoch || 0, totalSamples: data.totalSamples || 0, timestamp: data.timestamp || Date.now() });
      }
    }
    return checkpoints.sort((a, b) => b.timestamp - a.timestamp);
  } catch {
    return [];
  }
}, "fetchCheckpoints");

export const loadCheckpointQuery = query(async (filename) => {
  "use server";
  try {
    return { success: true, data: JSON.parse(await fs.readFile(path.join(VOLUME_CHECKPOINT_DIR, filename), "utf-8")) };
  } catch (err) {
    return { success: false, error: err.message };
  }
}, "loadCheckpoint");

export const triggerDownloadDaemon = action(async (formData) => {
  "use server";
  try {
    const response = await fetch("http://127.0.0.1:5000/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ n: parseInt(formData.get("n") || "5", 10) })
    });
    return { success: response.ok, httpStatus: response.status, ...(await response.json()) };
  } catch (err) {
    return { success: false, status: "error", message: `Daemon unreachable: ${err.message}` };
  }
});

export const fetchDaemonStatus = query(async () => {
  "use server";
  try {
    const res = await fetch("http://127.0.0.1:5000/status");
    return res.ok ? await res.json() : { state: "IDLE", progress: 0, total: 0, pending_articles: 0, current_sentence: "" };
  } catch {
    return { state: "OFFLINE", progress: 0, total: 0, pending_articles: 0, current_sentence: "" };
  }
}, "daemonStatus");

export function tokensToText(tokenIds, vocab) {
  const dict = vocab?.length ? vocab : ["transformer", "attention", "weights", "matrix", "vector"];
  return tokenIds.map(id => dict[Math.abs(Math.round(id)) % dict.length]).join(" ");
}

export function random_matrix(rows, cols, scale = 0.02) {
  const data = Array.from({ length: rows * cols }, () => (Math.random() * 2 - 1) * scale);
  return np.array(data, { dtype: np.float32 }).reshape([rows, cols]);
}

export function random_vector(dim, scale = 0.02) {
  const data = Array.from({ length: dim }, () => (Math.random() * 2 - 1) * scale);
  return np.array(data, { dtype: np.float32 });
}

export function layerNorm(x, eps = 1e-5) {
  const mean = x.ref.mean();
  const variance = x.ref.sub(mean.ref).square().mean();
  const normalized = x.ref.sub(mean.ref).div(variance.ref.add(eps).sqrt());
  mean.dispose();
  variance.dispose();
  return normalized;
}

export function softmax(arr) {
  const maxVal = arr.max().toArray()[0];
  const shifted = arr.ref.sub(maxVal);
  const exps = np.exp(shifted.ref);
  const sumExp = exps.ref.sum().toArray()[0] + 1e-9;
  return exps.div(sumExp);
}

export function multiHeadAttention(Q, K, V, numHeads = 16) {
  const dModel = Q.shape[0];
  const dHead = Math.floor(dModel / numHeads);
  const d_k = Math.sqrt(dHead) || 1;

  let headOutputs = [];
  for (let h = 0; h < numHeads; h++) {
    const start = h * dHead;
    const end = (h === numHeads - 1) ? dModel : (h + 1) * dHead;
    
    const Q_h = Q.ref.slice([start], [end]);
    const K_h = K.ref.slice([start], [end]);
    const V_h = V.ref.slice([start], [end]);

    const scores = Q_h.ref.matmul(K_h.ref.reshape([1, K_h.shape[0]])).div(d_k);
    const weights = softmax(scores);
    const headOut = V_h.ref.mul(weights.ref.toArray()[0] || 1.0);
    headOutputs.push(headOut);

    Q_h.dispose(); K_h.dispose(); V_h.dispose(); scores.dispose(); weights.dispose();
  }

  const concatenated = np.concatenate(headOutputs);
  headOutputs.forEach(t => t.dispose());
  return concatenated;
}

const batchProject = vmap((W, x) => W.matmul(x));

class ParameterStore {
  constructor(initial_params) {
    this.active_params = initial_params;
    this.inference_snapshot = {};
    for (const [k, v] of Object.entries(initial_params)) {
      this.inference_snapshot[k] = typeof v.copy === "function" ? v.copy() : v;
    }
    this.lock = false;
  }
  update(new_params) {
    this.active_params = new_params;
    if (!this.lock) {
      this.lock = true;
      for (const [k, v] of Object.entries(new_params)) {
        if (this.inference_snapshot[k] && typeof this.inference_snapshot[k].dispose === "function") {
          this.inference_snapshot[k].dispose();
        }
        this.inference_snapshot[k] = typeof v.copy === "function" ? v.copy() : v;
      }
      this.lock = false;
    }
  }
  getInferenceSnapshot() {
    const snap = {};
    for (const [k, v] of Object.entries(this.inference_snapshot)) {
      snap[k] = typeof v.copy === "function" ? v.copy() : v;
    }
    return snap;
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
      words: (item.content || "").replace(/[^\w\s]/gi, "").toLowerCase().split(/\s+/).filter(w => w.length > 0)
    };
  }

  sample_window() {
    const { docId, words } = this._getWords();
    const min_window = 40;
    const max_window = words.length;
    let window_size = max_window >= min_window ? Math.floor(Math.random() * (max_window - min_window + 1)) + min_window : max_window;
    let start_pos = max_window >= min_window ? Math.floor(Math.random() * (max_window - window_size + 1)) : 0;
    let sampled_span = words.slice(start_pos, start_pos + window_size);

    while (sampled_span.length < min_window) {
      sampled_span.push(this.vocab[Math.floor(Math.random() * this.vocab.length)] || "prediction");
    }

    const span = sampled_span.join(" ");
    const token_ids = sampled_span.map(w => {
      let hash = 0;
      for (let i = 0; i < w.length; i++) hash = (hash << 5) - hash + w.charCodeAt(i);
      return Math.abs(hash) % Math.max(1, this.vocab.length);
    });

    return { token_ids, span, charLength: span.length, docId };
  }

  sample_inference_seed() {
    const { docId, words } = this._getWords();
    const window_size = 5;
    let start_pos = words.length > window_size ? Math.floor(Math.random() * (words.length - window_size)) : 0;
    let sampled_span = words.slice(start_pos, start_pos + window_size);

    while (sampled_span.length < window_size) {
      sampled_span.push(this.vocab[Math.floor(Math.random() * this.vocab.length)] || "prediction");
    }

    const span = sampled_span.join(" ");
    const token_ids = sampled_span.map(w => {
      let hash = 0;
      for (let i = 0; i < w.length; i++) hash = (hash << 5) - hash + w.charCodeAt(i);
      return Math.abs(hash) % Math.max(1, this.vocab.length);
    });

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
    
    let token_ids;
    let docId = "custom_seed";
    if (seedTokenIds && seedTokenIds.length > 0) {
      token_ids = seedTokenIds;
    } else {
      const sampled = sampler.sample_inference_seed();
      token_ids = sampled.token_ids;
      docId = sampled.docId;
    }
    const seedDecoded = tokensToText(token_ids, this.vocab);
    
    const z_seed_arr = new Array(this.target_dim).fill(0);
    token_ids.forEach((id, idx) => {
      if (idx < this.target_dim) {
        z_seed_arr[idx] = (id / Math.max(1, this.vocab.length)) * 2.0 - 1.0;
      }
    });
    
    let current_z_arr = [...z_seed_arr];
    const mockPartsOfSpeech = ["n", "v", "adj", "adv", "pron", "det", "prep", "conj"];

    const sampleNextTokenId = (logitsArr) => {
      let scaled = logitsArr.map(l => l / Math.max(1e-5, temperature));
      let probs = softmax(np.array(scaled, { dtype: np.float32 })).toArray();
      let indexedProbs = probs.map((p, idx) => ({ p, idx }));
      
      if (topK > 0 && topK < indexedProbs.length) {
        indexedProbs.sort((a, b) => b.p - a.p);
        indexedProbs = indexedProbs.slice(0, topK);
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

    const trajectoryFull = [];
    const trajectoryHalf = [];

    for (let step = 1; step <= steps; step++) {
      const tVal = 1.0 - ((step - 1) / (steps - 1 || 1));
      const paddedArr = [...current_z_arr];
      const inputVec = np.array([...paddedArr, tVal], { dtype: np.float32 });
      
      const rawLogitsTensor = params.W_denoise.ref.matmul(inputVec);
      const rawLogits = rawLogitsTensor.toArray();
      rawLogitsTensor.dispose();
      inputVec.dispose();
      
      const alpha = 1.0 - tVal * 0.03;
      current_z_arr = current_z_arr.map((val, idx) => alpha * val + 0.05 * rawLogits[idx % rawLogits.length]);
      
      const truncatedTokenIds = [];
      for (let i = 0; i < targetCompletionTokens; i++) {
        const stepLogits = rawLogits.map((l, lIdx) => l + Math.sin(i + lIdx + tVal));
        const chosenId = sampleNextTokenId(stepLogits);
        truncatedTokenIds.push(chosenId);
      }

      const pureCompletion = tokensToText(truncatedTokenIds, this.vocab);
      const lexicalEntries = truncatedTokenIds.map((id, index) => {
        const lemma = this.vocab[id % this.vocab.length] || "term";
        const pos = mockPartsOfSpeech[(id + index) % mockPartsOfSpeech.length];
        return [lemma, pos];
      });

      const stepInfo = { step, t: Number(tVal.toFixed(2)), logits: rawLogits, pureCompletion, lexicalEntries };
      trajectoryFull.push(stepInfo);
      trajectoryHalf.push(stepInfo);
    }

    for (const v of Object.values(params)) v.dispose();
    return { seedDecoded, docId, trajectoryFull, trajectoryHalf };
  }

  trainStep(token_ids) {
    if (!this._previousLoss) this._previousLoss = null;

    const x_padded_arr = new Array(this.target_dim).fill(0);
    token_ids.forEach((id, idx) => { if (idx < this.target_dim) x_padded_arr[idx] = (id / this.vocab.length) * 2.0 - 1.0; });
    const x_padded = np.array(x_padded_arr, { dtype: np.float32 });

    const params = this.store.active_params;
    const { z_full, z_half, x_rec_full, x_rec_half } = this.encode(params, x_padded);

    const xRecFullArr = x_rec_full.toArray();
    const xRecHalfArr = x_rec_half.toArray();
    const zFullArr = z_full.toArray();

    let mse_full = 0, mse_half = 0;
    for (let i = 0; i < x_padded_arr.length; i++) {
      mse_full += Math.pow(x_padded_arr[i] - xRecFullArr[i], 2);
      mse_half += Math.pow(x_padded_arr[i] - xRecHalfArr[i], 2);
    }
    mse_full = Math.min(1, Math.max(0, mse_full / x_padded_arr.length));
    mse_half = Math.min(1, Math.max(0, mse_half / x_padded_arr.length));

    const lossDelta = this._previousLoss !== null ? this._previousLoss - mse_full : 0;
    const isConverging = lossDelta >= 0;
    this._previousLoss = mse_full;

    const lr = 0.001;
    const diff = x_rec_full.ref.sub(x_padded);
    const grad = diff.reshape([diff.shape[0], 1]).matmul(z_full.ref.reshape([1, z_full.shape[0]]));
    const updated_W_full = params.W_full.ref.sub(grad.mul(lr));

    const new_params = { ...params, W_full: updated_W_full };
    this.store.update(new_params);

    const decoded = tokensToText(zFullArr.map(v => Math.abs(Math.round((v + 1.0) * 0.5 * this.vocab.length))), this.vocab);

    x_padded.dispose(); z_full.dispose(); z_half.dispose(); x_rec_full.dispose(); x_rec_half.dispose(); diff.dispose(); grad.dispose();

    return {
      loss_full: mse_full,
      loss_half: mse_half,
      loss_delta: lossDelta,
      is_converging: isConverging,
      fidelity_gain_pct: Math.max(0, ((mse_half - mse_full) / (mse_half + 1e-6)) * 100),
      decoded
    };
  }
}

export default function Home() {
  const runDaemonAction = useAction(triggerDownloadDaemon);
  const executeSaveCheckpoint = useAction(saveCheckpointAction);

  const [vocab, setVocab] = createSignal(["transformer", "attention", "weights", "matrix", "vector"]);
  const [corpus, setCorpus] = createSignal([]);
  const [documentCount, setDocumentCount] = createSignal(0);
  const [isTraining, setIsTraining] = createSignal(false);
  const [epoch, setEpoch] = createSignal(0);
  const [totalSamples, setTotalSamples] = createSignal(0);
  const [lossFull, setLossFull] = createSignal(0.18);
  const [lossHalf, setLossHalf] = createSignal(0.42);
  const [fidelityGain, setFidelityGain] = createSignal(40.1);
  const [lossDelta, setLossDelta] = createSignal(0);
  const [isConverging, setIsConverging] = createSignal(true);
  const [sampleLogs, setSampleLogs] = createSignal([]);
  const [downloadStatus, setDownloadStatus] = createSignal("Idle");
  const [pendingArticlesCount, setPendingArticlesCount] = createSignal(0);
  const [currentProcessingSentence, setCurrentProcessingSentence] = createSignal("");
  const [articleCount, setArticleCount] = createSignal(5);
  const [isPollingDaemon, setIsPollingDaemon] = createSignal(false);

  const [checkpoints, setCheckpoints] = createSignal([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = createSignal("");
  const [lastAutoSave, setLastAutoSave] = createSignal("Never");
  const [isSaving, setIsSaving] = createSignal(false);

  const [diffusionResult, setDiffusionResult] = createSignal(null);
  const [isInferring, setIsInferring] = createSignal(false);
  const [targetTokens, setTargetTokens] = createSignal(16);
  const [temperature, setTemperature] = createSignal(0.8);
  const [topK, setTopK] = createSignal(50);
  const [topP, setTopP] = createSignal(0.9);

  const [prologStats, setPrologStats] = createSignal({ 
    totalEntries: 0, foundVocabCount: 0, coveragePct: 0, 
    topTenWords: [], foundPosCounts: {}, totalPosCounts: {}, 
    topTenSentenceFormats: [], topTenFlatPos: [], avgSentenceLength: 0.0, uniqueSentenceFormatsCount: 0, sampleEntries: [], status: "loading" 
  });

  const encoder = new TransformerDiffusionEncoder(2000, 2000, vocab());
  const sampler = new WindowSampler(vocab(), corpus());

  let trainTimer = null;
  let daemonPollTimer = null;

  const refreshData = async () => {
    const words = await fetchVocabQuery();
    if (words?.length) { setVocab(words); encoder.setVocab(words); sampler.setVocab(words); }
    const articles = await fetchArticleCorpusQuery();
    if (articles?.length) { setCorpus(articles); sampler.setCorpus(articles); }
    setDocumentCount(articles.length);
    const stats = await fetchPrologStatsQuery();
    if (stats) setPrologStats(stats);
    const statusData = await fetchDaemonStatus();
    if (statusData.pending_articles !== undefined) {
      setPendingArticlesCount(statusData.pending_articles);
    }
    if (statusData.current_sentence !== undefined) {
      setCurrentProcessingSentence(statusData.current_sentence);
    }
  };

  const refreshCheckpointList = async () => setCheckpoints(await fetchCheckpointsQuery());

  const performAutoSave = async () => {
    if (isSaving()) return;
    setIsSaving(true);
    const payload = { epoch: epoch(), totalSamples: totalSamples(), lossFull: lossFull(), lossHalf: lossHalf(), fidelityGain: fidelityGain(), params: encoder.store.active_params, timestamp: Date.now() };
    const formData = new FormData();
    formData.append("payload", JSON.stringify(payload));
    const res = await executeSaveCheckpoint(formData);
    if (res.success) { setLastAutoSave(`${res.name} (${res.timestamp})`); await refreshCheckpointList(); }
    setIsSaving(false);
  };

  const handleLoadCheckpoint = async () => {
    if (!selectedCheckpoint()) return;
    const res = await loadCheckpointQuery(selectedCheckpoint());
    if (res.success && res.data) {
      encoder.store.update(res.data.params);
      setEpoch(res.data.epoch || 0);
      setTotalSamples(res.data.totalSamples || 0);
      setLossFull(res.data.lossFull || 0.18);
      setLossHalf(res.data.lossHalf || 0.42);
      setFidelityGain(res.data.fidelityGain || 0);
    }
  };

  const stopDaemonPolling = () => { if (daemonPollTimer) clearInterval(daemonPollTimer); setIsPollingDaemon(false); };
  const startDaemonPolling = () => {
    stopDaemonPolling();
    setIsPollingDaemon(true);
    daemonPollTimer = setInterval(async () => {
      const statusData = await fetchDaemonStatus();
      if (statusData.state) {
        setDownloadStatus(`[${statusData.state}] Processed ${statusData.progress || 0}/${statusData.total || 0} (Pending: ${statusData.pending_articles || 0})`);
        if (statusData.pending_articles !== undefined) setPendingArticlesCount(statusData.pending_articles);
        if (statusData.current_sentence !== undefined) setCurrentProcessingSentence(statusData.current_sentence);
        await refreshData();
        if (["COMPLETED", "FAILED", "IDLE"].includes(statusData.state)) {
          await refreshData();
          if (statusData.state !== "PROCESSING") stopDaemonPolling();
        }
      }
    }, 1000);
  };

  const runTrainingStep = () => {
    const sampled = sampler.sample_window();
    const metrics = encoder.trainStep(sampled.token_ids);
    const nextSamples = totalSamples() + 1;
    setTotalSamples(nextSamples);

    if (nextSamples % 10 === 0) {
      setEpoch(e => {
        const nextEpoch = e + 1;
        if (nextEpoch % 10 === 0) performAutoSave();
        return nextEpoch;
      });
    }

    setLossFull(metrics.loss_full);
    setLossHalf(metrics.loss_half);
    setFidelityGain(metrics.fidelity_gain_pct);
    setLossDelta(metrics.loss_delta);
    setIsConverging(metrics.is_converging);

    setSampleLogs(prev => [
      { id: Date.now(), docId: sampled.docId, span: sampled.span.slice(0, 52) + (sampled.span.length > 52 ? "..." : ""), charLength: sampled.charLength, lossFull: metrics.loss_full.toFixed(4), lossHalf: metrics.loss_half.toFixed(4), tokenIds: sampled.token_ids },
      ...prev.slice(0, 19)
    ]);
  };

  const toggleTraining = () => {
    if (isTraining()) { setIsTraining(false); if (trainTimer) clearInterval(trainTimer); }
    else { setIsTraining(true); trainTimer = setInterval(runTrainingStep, 200); }
  };

  const runInference = (customSeedTokens = null) => {
    setIsInferring(true);
    setTimeout(() => {
      setDiffusionResult(encoder.diffusionInversionAndSample(32, sampler, targetTokens(), temperature(), topK(), topP(), customSeedTokens));
      setIsInferring(false);
    }, 50);
  };

  const handleDownloadSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("n", articleCount());
    setDownloadStatus("Triggering background daemon...");
    const res = await runDaemonAction(formData);
    if (res.success) startDaemonPolling();
    else setDownloadStatus(`Failed: ${res.message || "Unknown"}`);
  };

  onMount(async () => { await refreshData(); await refreshCheckpointList(); });
  onCleanup(() => { if (trainTimer) clearInterval(trainTimer); stopDaemonPolling(); });

  const cardStyle = { "background-color": "#0f172a", "border": "1px solid #1e293b", "border-radius": "10px", "padding": "28px", "display": "flex", "flex-direction": "column", "gap": "20px", "box-shadow": "0 25px 30px -5px rgba(0, 0, 0, 0.2)", "width": "100%", "box-sizing": "border-box" };
  const inputStyle = { "background-color": "#020617", "border": "1px solid #1e293b", "border-radius": "6px", "padding": "12px 16px", "color": "#f8fafc", "font-size": "1rem", "outline": "none" };

  return (
    <div style={{ "min-height": "100vh", "background-color": "#020617", "color": "#f8fafc", "padding": "32px", "font-family": "system-ui, -apple-system, sans-serif", "width": "100vw", "box-sizing": "border-box", "overflow-x": "hidden", "margin": "0", "font-size": "1.125rem", "line-height": "1.7" }}>
      <div style={{ "width": "100%", "max-width": "100%", "margin": "0 auto", "display": "flex", "flex-direction": "column", "gap": "32px", "box-sizing": "border-box" }}>
        
        {/* Header */}
        <div style={{ "border-bottom": "1px solid #1e293b", "padding-bottom": "24px", "display": "flex", "justify-content": "space-between", "align-items": "center", "width": "100%" }}>
          <div>
            <h1 style={{ "font-size": "2.25rem", "font-weight": "800", "color": "#34d399", "margin": "0", "letter-spacing": "-0.025em" }}>Transformer Attention & Diffusion Engine</h1>
            <p style={{ "font-size": "1.125rem", "color": "#94a3b8", "margin": "8px 0 0 0" }}>Next-Token Prediction & Diffusion Architecture with Karpathy Decoding</p>
          </div>
        </div>

        {/* Consolidated Top Row: Vocabulary / Prolog stats, Sentence Format Metrics & Wikipedia Downloader */}
        <div style={cardStyle}>
          <div style={{ "display": "flex", "justify-content": "space-between", "align-items": "center", "flex-wrap": "wrap", "gap": "16px", "width": "100%" }}>
            <div style={{ "display": "flex", "align-items": "center", "gap": "16px" }}>
              <h2 style={{ "font-size": "1.5rem", "font-weight": "700", "color": "#f8fafc", "margin": "0" }}>Vocabulary Engine & Wikipedia Ingestion</h2>
              <span style={{ "font-size": "0.875rem", "padding": "4px 12px", "border-radius": "6px", "background-color": "#020617", "color": "#34d399", "border": "1px solid #1e293b", "font-weight": "600" }}>
                Status: {prologStats().status}
              </span>
            </div>
            <div style={{ "display": "flex", "align-items": "center", "gap": "12px", "flex-wrap": "wrap" }}>
              <span style={{ "font-size": "0.875rem", "padding": "8px 16px", "border-radius": "6px", "background-color": "#020617", "color": "#cbd5e1", "border": "1px solid #1e293b" }}>
                Vocab Size: <strong style={{ "color": "#ffffff" }}>{prologStats().totalEntries || vocab().length}</strong>
              </span>
              <span style={{ "font-size": "0.875rem", "padding": "8px 16px", "border-radius": "6px", "background-color": "#020617", "color": "#cbd5e1", "border": "1px solid #1e293b" }}>
                Found in Articles: <strong style={{ "color": "#34d399" }}>{prologStats().foundVocabCount}</strong> ({prologStats().coveragePct}%)
              </span>
              <span style={{ "font-size": "0.875rem", "padding": "8px 16px", "border-radius": "6px", "background-color": "#020617", "color": "#cbd5e1", "border": "1px solid #1e293b" }}>
                Pending Analysis: <strong style={{ "color": "#f59e0b" }}>{pendingArticlesCount()}</strong>
              </span>
              <span style={{ "font-size": "0.875rem", "padding": "8px 16px", "border-radius": "6px", "background-color": "#020617", "color": "#cbd5e1", "border": "1px solid #1e293b" }}>
                Unique Sentence Formats: <strong style={{ "color": "#38bdf8" }}>{prologStats().uniqueSentenceFormatsCount}</strong>
              </span>
              <span style={{ "font-size": "0.875rem", "padding": "8px 16px", "border-radius": "6px", "background-color": "#020617", "color": "#cbd5e1", "border": "1px solid #1e293b" }}>
                Avg Sentence Length: <strong style={{ "color": "#38bdf8" }}>{prologStats().avgSentenceLength} words</strong>
              </span>
              <span style={{ "font-size": "0.875rem", "padding": "8px 16px", "border-radius": "6px", "background-color": "#020617", "color": "#cbd5e1", "border": "1px solid #1e293b" }}>
                Document Count: <strong style={{ "color": "#ffffff" }}>{documentCount()}</strong>
              </span>
            </div>
          </div>

          <div style={{ "display": "grid", "grid-template-columns": "repeat(auto-fit, minmax(320px, 1fr))", "gap": "20px", "align-items": "start", "width": "100%" }}>
            
            {/* Top Ten Words Frequency Table */}
            <div style={{ "background-color": "#020617", "padding": "16px", "border-radius": "6px", "border": "1px solid #1e293b", "display": "flex", "flex-direction": "column", "gap": "10px" }}>
              <div style={{ "font-size": "0.95rem", "font-weight": "700", "color": "#34d399" }}>Top Ten Vocabulary Words Frequency</div>
              {prologStats().topTenWords?.length > 0 ? (
                <div style={{ "display": "flex", "flex-direction": "column", "gap": "4px", "max-height": "160px", "overflow-y": "auto" }}>
                  <For each={prologStats().topTenWords}>
                    {(item) => (
                      <div style={{ "display": "flex", "justify-content": "space-between", "font-size": "0.85rem", "font-family": "monospace", "color": "#cbd5e1", "border-bottom": "1px solid #0f172a", "padding": "2px 0" }}>
                        <span>{item.word}</span>
                        <span style={{ "color": "#38bdf8" }}>{item.count} hits</span>
                      </div>
                    )}
                  </For>
                </div>
              ) : (
                <div style={{ "font-size": "0.85rem", "color": "#64748b" }}>No word frequencies recorded yet. Run a download job!</div>
              )}
            </div>

            {/* Top Ten Sentence Formats Table */}
            <div style={{ "background-color": "#020617", "padding": "16px", "border-radius": "6px", "border": "1px solid #1e293b", "display": "flex", "flex-direction": "column", "gap": "10px" }}>
              <div style={{ "font-size": "0.95rem", "font-weight": "700", "color": "#34d399" }}>Top Ten Sentence Formats (Hierarchical DCG)</div>
              {prologStats().topTenSentenceFormats?.length > 0 ? (
                <div style={{ "display": "flex", "flex-direction": "column", "gap": "4px", "max-height": "160px", "overflow-y": "auto" }}>
                  <For each={prologStats().topTenSentenceFormats}>
                    {(item) => (
                      <div style={{ "display": "flex", "justify-content": "space-between", "font-size": "0.85rem", "font-family": "monospace", "color": "#cbd5e1", "border-bottom": "1px solid #0f172a", "padding": "2px 0" }}>
                        <span style={{ "color": "#38bdf8", "max-width": "70%", "overflow": "hidden", "text-overflow": "ellipsis" }}>{item.format}</span>
                        <span>{item.count} hits</span>
                      </div>
                    )}
                  </For>
                </div>
              ) : (
                <div style={{ "font-size": "0.85rem", "color": "#64748b" }}>No sentence format metrics analyzed yet.</div>
              )}
            </div>

            {/* Top Ten Sentences by Parts of Speech (Flat POS) */}
            <div style={{ "background-color": "#020617", "padding": "16px", "border-radius": "6px", "border": "1px solid #1e293b", "display": "flex", "flex-direction": "column", "gap": "10px" }}>
              <div style={{ "font-size": "0.95rem", "font-weight": "700", "color": "#34d399" }}>Top Ten Sentences by Parts of Speech (Flat POS)</div>
              {prologStats().topTenFlatPos?.length > 0 ? (
                <div style={{ "display": "flex", "flex-direction": "column", "gap": "4px", "max-height": "160px", "overflow-y": "auto" }}>
                  <For each={prologStats().topTenFlatPos}>
                    {(item) => (
                      <div style={{ "display": "flex", "justify-content": "space-between", "font-size": "0.85rem", "font-family": "monospace", "color": "#cbd5e1", "border-bottom": "1px solid #0f172a", "padding": "2px 0" }}>
                        <span style={{ "color": "#38bdf8", "max-width": "70%", "overflow": "hidden", "text-overflow": "ellipsis" }}>{item.format}</span>
                        <span>{item.count} hits</span>
                      </div>
                    )}
                  </For>
                </div>
              ) : (
                <div style={{ "font-size": "0.85rem", "color": "#64748b" }}>No flat POS metrics analyzed yet.</div>
              )}
            </div>

            {/* Downloader & Live Parts of Speech */}
            <div style={{ "display": "flex", "flex-direction": "column", "gap": "16px" }}>
              <form onSubmit={handleDownloadSubmit} style={{ "display": "flex", "gap": "12px", "align-items": "center", "margin": "0", "width": "100%" }}>
                <input 
                  type="number" 
                  min="1" 
                  max="100" 
                  value={articleCount()} 
                  onInput={(e) => setArticleCount(parseInt(e.target.value) || 5)}
                  style={{ ...inputStyle, "width": "100px" }}
                />
                <button type="submit" style={{ "padding": "12px 24px", "background-color": "#2563eb", "color": "#ffffff", "border-radius": "6px", "font-weight": "600", "font-size": "1rem", "border": "none", "cursor": "pointer", "white-space": "nowrap" }}>
                  Fetch Wikipedia
                </button>
                <div style={{ "font-size": "0.875rem", "color": "#94a3b8", "overflow": "hidden", "text-overflow": "ellipsis", "white-space": "nowrap", "flex": "1" }}>
                  {downloadStatus()}
                </div>
              </form>

              {/* Live Processing Sentence View */}
              <div style={{ "background-color": "#020617", "padding": "12px 16px", "border-radius": "6px", "border": "1px solid #1e293b", "display": "flex", "flex-direction": "column", "gap": "6px" }}>
                <div style={{ "font-size": "0.85rem", "font-weight": "600", "color": "#34d399" }}>Currently Processing Sentence:</div>
                <div style={{ "font-size": "0.85rem", "color": "#f8fafc", "font-family": "monospace", "background": "#0f172a", "padding": "8px", "border-radius": "4px", "border": "1px solid #1e293b", "word-break": "break-all", "max-height": "80px", "overflow-y": "auto" }}>
                  {currentProcessingSentence() || "Idle / Waiting for task..."}
                </div>
              </div>

              <div style={{ "background-color": "#020617", "padding": "12px 16px", "border-radius": "6px", "border": "1px solid #1e293b", "display": "flex", "flex-direction": "column", "gap": "6px" }}>
                <div style={{ "font-size": "0.85rem", "font-weight": "600", "color": "#94a3b8" }}>Live Parts of Speech Distribution (Found / Total):</div>
                <div style={{ "display": "flex", "flex-wrap": "wrap", "gap": "8px" }}>
                  <For each={Object.keys(prologStats().totalPosCounts || { n: 0, v: 0, adj: 0, adv: 0 })}>
                    {(posKey) => {
                      const foundVal = prologStats().foundPosCounts?.[posKey] || 0;
                      const totalVal = prologStats().totalPosCounts?.[posKey] || 0;
                      return (
                        <span style={{ "font-size": "0.75rem", "background-color": "#0f172a", "border": "1px solid #1e293b", "padding": "4px 8px", "border-radius": "4px", "color": "#cbd5e1", "font-family": "monospace" }}>
                          <strong style={{ "color": "#34d399" }}>{posKey}</strong>: {foundVal}/{totalVal}
                        </span>
                      );
                    }}
                  </For>
                </div>
              </div>

            </div>

          </div>
        </div>

        {/* Training Daemon & Checkpoint Grid */}
        <div style={{ "display": "grid", "grid-template-columns": "repeat(auto-fit, minmax(450px, 1fr))", "gap": "32px", "width": "100%" }}>
          
          <div style={cardStyle}>
            <div style={{ "display": "flex", "justify-content": "space-between", "align-items": "center", "width": "100%" }}>
              <div style={{ "display": "flex", "align-items": "center", "gap": "12px" }}>
                <h2 style={{ "font-size": "1.5rem", "font-weight": "700", "color": "#f8fafc", "margin": "0" }}>Model Training Daemon</h2>
                <span style={{ "font-size": "0.75rem", "padding": "4px 10px", "border-radius": "9999px", "background-color": isConverging() ? "#064e3b" : "#7f1d1d", "color": isConverging() ? "#34d399" : "#fca5a5", "border": `1px solid ${isConverging() ? "#059669" : "#dc2626"}`, "font-weight": "600" }}>
                  {isConverging() ? "● Converging Active" : "○ Adjusting..."}
                </span>
              </div>
              <button onClick={toggleTraining} style={{ "padding": "12px 24px", "border-radius": "6px", "font-weight": "600", "font-size": "1rem", "cursor": "pointer", "border": "none", "color": "#ffffff", "background-color": isTraining() ? "#e11d48" : "#059669" }}>
                {isTraining() ? "Pause" : "Start"}
              </button>
            </div>

            <div style={{ "display": "grid", "grid-template-columns": "repeat(3, minmax(0, 1fr))", "gap": "16px", "width": "100%" }}>
              <div style={{ "background-color": "#020617", "padding": "16px", "border-radius": "6px", "border": "1px solid #1e293b" }}>
                <div style={{ "font-size": "0.875rem", "color": "#94a3b8" }}>Epoch</div>
                <div style={{ "font-size": "1.75rem", "font-weight": "800", "color": "#f8fafc", "margin-top": "6px" }}>{epoch()}</div>
              </div>
              <div style={{ "background-color": "#020617", "padding": "16px", "border-radius": "6px", "border": "1px solid #1e293b" }}>
                <div style={{ "font-size": "0.875rem", "color": "#94a3b8" }}>Samples</div>
                <div style={{ "font-size": "1.75rem", "font-weight": "800", "color": "#f8fafc", "margin-top": "6px" }}>{totalSamples()}</div>
              </div>
              <div style={{ "background-color": "#020617", "padding": "16px", "border-radius": "6px", "border": "1px solid #1e293b" }}>
                <div style={{ "font-size": "0.875rem", "color": "#94a3b8" }}>Gain</div>
                <div style={{ "font-size": "1.75rem", "font-weight": "800", "color": "#34d399", "margin-top": "6px" }}>{fidelityGain().toFixed(1)}%</div>
              </div>
            </div>

            <div style={{ "display": "flex", "flex-direction": "column", "gap": "14px", "background-color": "#020617", "padding": "16px", "border-radius": "8px", "border": "1px solid #1e293b", "width": "100%", "box-sizing": "border-box" }}>
              <div style={{ "display": "flex", "flex-direction": "column", "gap": "6px" }}>
                <div style={{ "display": "flex", "justify-content": "space-between", "font-size": "0.875rem", "color": "#94a3b8" }}>
                  <span>Full Loss (Δ: {lossDelta >= 0 ? `-${Math.abs(lossDelta).toFixed(4)}` : `+${Math.abs(lossDelta).toFixed(4)}`})</span>
                  <span style={{ "color": "#34d399", "font-weight": "700" }}>{lossFull().toFixed(4)}</span>
                </div>
                <div style={{ "width": "100%", "background-color": "#0f172a", "border-radius": "9999px", "height": "8px", "border": "1px solid #1e293b", "overflow": "hidden" }}>
                  <div style={{ "background-color": "#10b981", "height": "100%", "width": `${Math.min(100, lossFull() * 100)}%` }}></div>
                </div>
              </div>
              <div style={{ "display": "flex", "flex-direction": "column", "gap": "6px" }}>
                <div style={{ "display": "flex", "justify-content": "space-between", "font-size": "0.875rem", "color": "#94a3b8" }}>
                  <span>Half Loss</span><span style={{ "color": "#3b82f6", "font-weight": "700" }}>{lossHalf().toFixed(4)}</span>
                </div>
                <div style={{ "width": "100%", "background-color": "#0f172a", "border-radius": "9999px", "height": "8px", "border": "1px solid #1e293b", "overflow": "hidden" }}>
                  <div style={{ "background-color": "#3b82f6", "height": "100%", "width": `${Math.min(100, lossHalf() * 100)}%` }}></div>
                </div>
              </div>
            </div>

            {/* Live Training Sample Activity Stream */}
            <div style={{ "background-color": "#020617", "padding": "14px", "border-radius": "6px", "border": "1px solid #1e293b", "display": "flex", "flex-direction": "column", "gap": "8px" }}>
              <div style={{ "font-size": "0.85rem", "font-weight": "700", "color": "#38bdf8" }}>Live Training Sample Activity (Click to Infeers Seed)</div>
              <div style={{ "display": "flex", "flex-direction": "column", "gap": "4px", "max-height": "140px", "overflow-y": "auto" }}>
                <For each={sampleLogs()}>
                  {(log) => (
                    <div 
                      onClick={() => runInference(log.tokenIds)}
                      style={{ "display": "flex", "justify-content": "space-between", "align-items": "center", "font-size": "0.8rem", "font-family": "monospace", "color": "#cbd5e1", "background": "#0f172a", "padding": "6px 8px", "border-radius": "4px", "border": "1px solid #1e293b", "cursor": "pointer" }}
                      title="Click to run inference using this trained sample sub-window!"
                    >
                      <span style={{ "overflow": "hidden", "text-overflow": "ellipsis", "white-space": "nowrap", "max-width": "70%" }}>[{log.docId}] {log.span}</span>
                      <span style={{ "color": "#34d399", "flex-shrink": "0" }}>loss: {log.lossFull}</span>
                    </div>
                  )}
                </For>
              </div>
            </div>

          </div>

          <div style={cardStyle}>
            <div style={{ "display": "flex", "justify-content": "space-between", "align-items": "center", "width": "100%" }}>
              <h2 style={{ "font-size": "1.5rem", "font-weight": "700", "color": "#f8fafc", "margin": "0" }}>Checkpoint Management</h2>
              <div style={{ "font-size": "0.875rem", "color": "#94a3b8" }}>Auto-Save: {lastAutoSave()}</div>
            </div>
            <div style={{ "display": "flex", "flex-direction": "column", "gap": "16px", "width": "100%" }}>
              <div style={{ "display": "flex", "gap": "12px", "width": "100%" }}>
                <select value={selectedCheckpoint()} onChange={(e) => setSelectedCheckpoint(e.target.value)} style={{ ...inputStyle, "flex": "1" }}>
                  <option value="">-- Select Checkpoint --</option>
                  <For each={checkpoints()}>{(c) => <option value={c.id}>{c.name} (Ep {c.epoch})</option>}</For>
                </select>
                <button onClick={handleLoadCheckpoint} style={{ "padding": "12px 24px", "background-color": "#334155", "color": "#ffffff", "border-radius": "6px", "font-size": "1rem", "border": "none", "cursor": "pointer", "font-weight": "600" }}>Load</button>
              </div>
              <button onClick={performAutoSave} disabled={isSaving()} style={{ "width": "100%", "padding": "12px 24px", "background-color": "#059669", "color": "#ffffff", "border-radius": "6px", "font-size": "1rem", "border": "none", "cursor": "pointer", "font-weight": "600" }}>
                {isSaving() ? "Saving Checkpoint..." : "Save Checkpoint Now"}
              </button>
            </div>
          </div>

        </div>

        {/* Karpathy nanoGPT Decoding Inference Section */}
        <div style={cardStyle}>
          <div style={{ "display": "flex", "justify-content": "space-between", "align-items": "center", "flex-wrap": "wrap", "gap": "16px", "width": "100%" }}>
            <div>
              <h2 style={{ "font-size": "1.5rem", "font-weight": "700", "color": "#f8fafc", "margin": "0" }}>Inference & nanoGPT Decoding Sandbox</h2>
              <p style={{ "font-size": "0.95rem", "color": "#94a3b8", "margin": "4px 0 0 0" }}>Test generation using Temperature, Top-K, and Top-P (Nucleus) sampling</p>
            </div>
            <button onClick={() => runInference(null)} disabled={isInferring()} style={{ "padding": "12px 28px", "background-color": "#2563eb", "color": "#ffffff", "border-radius": "6px", "font-weight": "600", "font-size": "1rem", "border": "none", "cursor": "pointer" }}>
              {isInferring() ? "Sampling..." : "Run Inference"}
            </button>
          </div>

          <div style={{ "display": "grid", "grid-template-columns": "repeat(auto-fit, minmax(200px, 1fr))", "gap": "16px", "width": "100%" }}>
            <div style={{ "background-color": "#020617", "padding": "12px", "border-radius": "6px", "border": "1px solid #1e293b", "display": "flex", "flex-direction": "column", "gap": "6px" }}>
              <label style={{ "font-size": "0.85rem", "color": "#94a3b8" }}>Temperature: <strong style={{ "color": "#38bdf8" }}>{temperature()}</strong></label>
              <input type="range" min="0.1" max="2.0" step="0.1" value={temperature()} onInput={(e) => setTemperature(parseFloat(e.target.value))} style={{ "width": "100%" }} />
            </div>
            <div style={{ "background-color": "#020617", "padding": "12px", "border-radius": "6px", "border": "1px solid #1e293b", "display": "flex", "flex-direction": "column", "gap": "6px" }}>
              <label style={{ "font-size": "0.85rem", "color": "#94a3b8" }}>Top-K: <strong style={{ "color": "#38bdf8" }}>{topK()}</strong></label>
              <input type="range" min="0" max="100" step="5" value={topK()} onInput={(e) => setTopK(parseInt(e.target.value))} style={{ "width": "100%" }} />
            </div>
            <div style={{ "background-color": "#020617", "padding": "12px", "border-radius": "6px", "border": "1px solid #1e293b", "display": "flex", "flex-direction": "column", "gap": "6px" }}>
              <label style={{ "font-size": "0.85rem", "color": "#94a3b8" }}>Top-P (Nucleus): <strong style={{ "color": "#38bdf8" }}>{topP()}</strong></label>
              <input type="range" min="0.1" max="1.0" step="0.05" value={topP()} onInput={(e) => setTopP(parseFloat(e.target.value))} style={{ "width": "100%" }} />
            </div>
            <div style={{ "background-color": "#020617", "padding": "12px", "border-radius": "6px", "border": "1px solid #1e293b", "display": "flex", "flex-direction": "column", "gap": "6px" }}>
              <label style={{ "font-size": "0.85rem", "color": "#94a3b8" }}>Target Tokens: <strong style={{ "color": "#38bdf8" }}>{targetTokens()}</strong></label>
              <input type="number" min="4" max="64" value={targetTokens()} onInput={(e) => setTargetTokens(parseInt(e.target.value) || 16)} style={{ ...inputStyle, "padding": "6px 10px", "font-size": "0.9rem" }} />
            </div>
          </div>

          {diffusionResult() && (
            <div style={{ "display": "flex", "flex-direction": "column", "gap": "16px", "background-color": "#020617", "padding": "20px", "border-radius": "8px", "border": "1px solid #1e293b" }}>
              <div style={{ "display": "flex", "justify-content": "space-between", "font-size": "0.9rem", "color": "#94a3b8" }}>
                <span>Seed Document: <strong style={{ "color": "#f8fafc" }}>{diffusionResult().docId}</strong></span>
                <span>Seed Prompt: <strong style={{ "color": "#34d399" }}>{diffusionResult().seedDecoded}</strong></span>
              </div>
              <div style={{ "font-family": "monospace", "font-size": "1rem", "color": "#e2e8f0", "background": "#0f172a", "padding": "16px", "border-radius": "6px", "border": "1px solid #1e293b", "word-break": "break-all" }}>
                <strong style={{ "color": "#38bdf8" }}>Decoded Output:</strong> {diffusionResult().trajectoryFull?.[diffusionResult().trajectoryFull.length - 1]?.pureCompletion}
              </div>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}