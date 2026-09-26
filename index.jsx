import { createSignal, For, onCleanup, onMount, Switch, Match, createEffect } from "solid-js";
import { useAction } from "@solidjs/router";
import { 
  fetchVocabQuery, 
  fetchArticleCorpusQuery, 
  fetchPrologStatsQuery, 
  fetchPaginatedArticlesQuery, 
  saveCheckpointAction, 
  loadCheckpointQuery, 
  triggerDownloadDaemon, 
  fetchDaemonStatus,
  performAutoSave,
  handleLoadCheckpoint,
  startDaemonPolling,
  stopDaemonPolling,
  TransformerDiffusionEncoder, 
  WindowSampler 
} from "../ml.js";

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

  const [prologStats, setPrologStats] = createSignal({ totalEntries: 0, foundVocabCount: 0, coveragePct: 0, topTenWords: [], foundPosCounts: {}, totalPosCounts: {}, sampleEntries: [] });

  const [activeTab, setActiveTab] = createSignal("vocabulary");
  const [vocabSubTab, setVocabSubTab] = createSignal("topWords");
  const [searchQuery, setSearchQuery] = createSignal("");
  const [currentPage, setCurrentPage] = createSignal(1);

  const [articleData, setArticleData] = createSignal({ items: [], total: 0, page: 1, limit: 5, pages: 1 });

  const loadArticlesAsync = (q, p) => {
    fetchPaginatedArticlesQuery(q, p).then(res => setArticleData(res));
  };

  createEffect(() => {
    loadArticlesAsync(searchQuery(), currentPage());
  });

  const encoder = new TransformerDiffusionEncoder(2000, 2000, vocab());
  const sampler = new WindowSampler(vocab(), corpus());

  let trainTimer = null;
  let daemonPollTimer = null;

  const refreshData = () => Promise.all([fetchVocabQuery(), fetchArticleCorpusQuery(), fetchPrologStatsQuery()]).then(([words, articles, stats]) => {
    if (words?.length) { setVocab(words); encoder.setVocab(words); sampler.setVocab(words); }
    if (articles?.length) { setCorpus(articles); sampler.setCorpus(articles); setDocumentCount(articles.length); }
    if (stats) setPrologStats(stats);
    loadArticlesAsync(searchQuery(), currentPage());
  });

  const refreshCheckpointList = () => fetchCheckpointsQuery().then(setCheckpoints);

  const triggerAutoSave = () => performAutoSave(isSaving, setIsSaving, epoch, totalSamples, lossFull, lossHalf, fidelityGain, encoder, setLastAutoSave, refreshCheckpointList, executeSaveCheckpoint);
  const triggerLoadCheckpoint = () => handleLoadCheckpoint(selectedCheckpoint, loadCheckpointQuery, encoder, setEpoch, setTotalSamples, setLossFull, setLossHalf, setFidelityGain);

  const triggerStartPolling = () => startDaemonPolling(daemonPollTimer, setIsPollingDaemon, fetchDaemonStatus, setDownloadStatus, refreshData, stopDaemonPolling);
  const triggerStopPolling = () => stopDaemonPolling(daemonPollTimer, setIsPollingDaemon);

  const runTrainingStep = () => {
    const sampled = sampler.sample_window();
    const metrics = encoder.trainStep(sampled.token_ids);
    const nextSamples = totalSamples() + 1;
    setTotalSamples(nextSamples);

    if (nextSamples % 10 === 0) {
      setEpoch(e => {
        const nextEpoch = e + 1;
        if (nextEpoch % 10 === 0) triggerAutoSave();
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

  const handleDownloadSubmit = (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("n", articleCount());
    setDownloadStatus("Triggering background daemon...");
    runDaemonAction(formData).then(res => res.success ? triggerStartPolling() : setDownloadStatus(`Failed: ${res.message || "Unknown"}`));
  };

  onMount(() => { refreshData(); refreshCheckpointList(); });
  onCleanup(() => { if (trainTimer) clearInterval(trainTimer); triggerStopPolling(); });

  const cardStyle = { "background-color": "#ffffff", "border": "1px solid #cbd5e1", "border-radius": "10px", "padding": "28px", "display": "flex", "flex-direction": "column", "gap": "20px", "box-shadow": "0 10px 15px -3px rgba(0, 0, 0, 0.05)", "width": "100%", "box-sizing": "border-box" };
  const inputStyle = { "background-color": "#f8fafc", "border": "1px solid #cbd5e1", "border-radius": "6px", "padding": "12px 16px", "color": "#0f172a", "font-size": "1rem", "outline": "none" };
  const tableHeaderStyle = { "padding": "10px 12px", "text-align": "left", "font-size": "0.85rem", "color": "#475569", "border-bottom": "1px solid #cbd5e1", "font-weight": "600" };
  const tableCellStyle = { "padding": "10px 12px", "font-size": "0.85rem", "color": "#334155", "border-bottom": "1px solid #f1f5f9", "font-family": "monospace" };
  const tabButtonStyle = (isActive) => ({ "padding": "10px 20px", "background-color": isActive ? "#2563eb" : "#f1f5f9", "color": isActive ? "#ffffff" : "#334155", "border": "1px solid #cbd5e1", "border-radius": "6px", "font-weight": "600", "cursor": "pointer", "display": "flex", "align-items": "center", "gap": "8px", "font-size": "0.95rem" });
  const subTabButtonStyle = (isActive) => ({ "padding": "6px 14px", "background-color": isActive ? "#0284c7" : "#e2e8f0", "color": isActive ? "#ffffff" : "#334155", "border": "none", "border-radius": "4px", "font-weight": "600", "cursor": "pointer", "font-size": "0.85rem" });
  const badgeStyle = { "background-color": "#e2e8f0", "color": "#0f172a", "padding": "2px 8px", "border-radius": "9999px", "font-size": "0.75rem", "font-weight": "bold" };

  return (
    <div style={{ "min-height": "100vh", "background-color": "#f8fafc", "color": "#0f172a", "padding": "32px", "font-family": "system-ui, -apple-system, sans-serif", "width": "100vw", "box-sizing": "border-box", "overflow-x": "hidden", "margin": "0", "font-size": "1.125rem", "line-height": "1.7" }}>
      <div style={{ "width": "100%", "max-width": "100%", "margin": "0 auto", "display": "flex", "flex-direction": "column", "gap": "32px", "box-sizing": "border-box" }}>
        
        <div style={{ "border-bottom": "1px solid #cbd5e1", "padding-bottom": "24px", "display": "flex", "justify-content": "space-between", "align-items": "center", "width": "100%" }}>
          <div>
            <h1 style={{ "font-size": "2.25rem", "font-weight": "800", "color": "#059669", "margin": "0", "letter-spacing": "-0.025em" }}>Transformer Attention & Diffusion Engine</h1>
            <p style={{ "font-size": "1.125rem", "color": "#64748b", "margin": "8px 0 0 0" }}>Next-Token Prediction & Diffusion Architecture with Karpathy Decoding</p>
          </div>
        </div>

        <div style={cardStyle}>
          <div style={{ "display": "flex", "justify-content": "space-between", "align-items": "center", "flex-wrap": "wrap", "gap": "16px", "width": "100%", "border-bottom": "1px solid #cbd5e1", "padding-bottom": "16px" }}>
            <div style={{ "display": "flex", "gap": "12px", "flex-wrap": "wrap" }}>
              <button onClick={() => setActiveTab("vocabulary")} style={tabButtonStyle(activeTab() === "vocabulary")}>
                Vocabulary Engine
                <span style={badgeStyle}>{prologStats().totalEntries || vocab().length}</span>
              </button>
              <button onClick={() => setActiveTab("corpus")} style={tabButtonStyle(activeTab() === "corpus")}>
                Corpus Search
                <span style={badgeStyle}>{articleData().total}</span>
              </button>
              <button onClick={() => setActiveTab("ingestion")} style={tabButtonStyle(activeTab() === "ingestion")}>
                Fetch Wikipedia Feed
                <span style={badgeStyle}>{documentCount()}</span>
              </button>
            </div>
          </div>

          <Switch>
            <Match when={activeTab() === "vocabulary"}>
              <div style={{ "display": "flex", "flex-direction": "column", "gap": "20px", "width": "100%" }}>
                <div style={{ "display": "flex", "justify-content": "space-between", "align-items": "center", "flex-wrap": "wrap", "gap": "16px" }}>
                  <table style={{ "border-collapse": "collapse", "background-color": "#ffffff", "border": "1px solid #cbd5e1", "border-radius": "6px", "overflow": "hidden" }}>
                    <thead>
                      <tr>
                        <th style={tableHeaderStyle}>Vocab Size</th>
                        <th style={tableHeaderStyle}>Found in Articles</th>
                        <th style={tableHeaderStyle}>Coverage</th>
                        <th style={tableHeaderStyle}>Documents</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td style={{ ...tableCellStyle, "color": "#0f172a", "font-weight": "bold" }}>{prologStats().totalEntries || vocab().length}</td>
                        <td style={{ ...tableCellStyle, "color": "#059669", "font-weight": "bold" }}>{prologStats().foundVocabCount}</td>
                        <td style={{ ...tableCellStyle, "color": "#0284c7" }}>{prologStats().coveragePct}%</td>
                        <td style={{ ...tableCellStyle, "color": "#0f172a" }}>{documentCount()}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <div style={{ "background-color": "#ffffff", "padding": "16px", "border-radius": "6px", "border": "1px solid #cbd5e1", "display": "flex", "flex-direction": "column", "gap": "12px", "box-shadow": "0 1px 3px rgba(0,0,0,0.05)", "width": "100%" }}>
                  <div style={{ "display": "flex", "justify-content": "space-between", "align-items": "center" }}>
                    <div style={{ "font-size": "0.95rem", "font-weight": "700", "color": "#0f172a" }}>
                      {vocabSubTab() === "topWords" ? "Top Ten Vocabulary Words Frequency" : "Live Parts of Speech Distribution"}
                    </div>
                    <div style={{ "display": "flex", "gap": "6px" }}>
                      <button onClick={() => setVocabSubTab("topWords")} style={subTabButtonStyle(vocabSubTab() === "topWords")}>Top 10</button>
                      <button onClick={() => setVocabSubTab("pos")} style={subTabButtonStyle(vocabSubTab() === "pos")}>Parts of Speech</button>
                    </div>
                  </div>

                  <Switch>
                    <Match when={vocabSubTab() === "topWords"}>
                      {prologStats().topTenWords?.length > 0 ? (
                        <div style={{ "max-height": "220px", "overflow-y": "auto" }}>
                          <table style={{ "width": "100%", "border-collapse": "collapse" }}>
                            <thead>
                              <tr>
                                <th style={tableHeaderStyle}>Word</th>
                                <th style={{ ...tableHeaderStyle, "text-align": "right" }}>Hits</th>
                              </tr>
                            </thead>
                            <tbody>
                              <For each={prologStats().topTenWords}>
                                {(item) => (
                                  <tr>
                                    <td style={tableCellStyle}>{item.word}</td>
                                    <td style={{ ...tableCellStyle, "text-align": "right", "color": "#0284c7" }}>{item.count}</td>
                                  </tr>
                                )}
                              </For>
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <div style={{ "font-size": "0.85rem", "color": "#64748b" }}>No word frequencies recorded yet. Run a download job!</div>
                      )}
                    </Match>
                    <Match when={vocabSubTab() === "pos"}>
                      <div style={{ "max-height": "220px", "overflow-y": "auto" }}>
                        <table style={{ "width": "100%", "border-collapse": "collapse" }}>
                          <thead>
                            <tr>
                              <th style={tableHeaderStyle}>Part of Speech</th>
                              <th style={{ ...tableHeaderStyle, "text-align": "right" }}>Found / Total</th>
                            </tr>
                          </thead>
                          <tbody>
                            <For each={Object.keys(prologStats().totalPosCounts || { n: 0, v: 0, adj: 0, adv: 0 })}>
                              {(posKey) => {
                                const foundVal = prologStats().foundPosCounts?.[posKey] || 0;
                                const totalVal = prologStats().totalPosCounts?.[posKey] || 0;
                                return (
                                  <tr>
                                    <td style={{ ...tableCellStyle, "color": "#059669", "font-weight": "bold" }}>{posKey}</td>
                                    <td style={{ ...tableCellStyle, "text-align": "right" }}>{foundVal} / {totalVal}</td>
                                  </tr>
                                );
                              }}
                            </For>
                          </tbody>
                        </table>
                      </div>
                    </Match>
                  </Switch>
                </div>
              </div>
            </Match>

            <Match when={activeTab() === "corpus"}>
              <div style={{ "display": "flex", "flex-direction": "column", "gap": "20px", "width": "100%" }}>
                <div style={{ "display": "flex", "justify-content": "space-between", "align-items": "center", "flex-wrap": "wrap", "gap": "16px", "width": "100%" }}>
                  <h3 style={{ "font-size": "1.2rem", "font-weight": "700", "color": "#059669", "margin": "0" }}>Article Corpus Full-Text & Semantic Embeddings Search</h3>
                  <input 
                    type="text" 
                    placeholder="Search downloaded articles..." 
                    value={searchQuery()} 
                    onInput={(e) => { setSearchQuery(e.target.value); setCurrentPage(1); }}
                    style={{ ...inputStyle, "width": "300px" }}
                  />
                </div>

                <div style={{ "overflow-x": "auto" }}>
                  <table style={{ "width": "100%", "border-collapse": "collapse" }}>
                    <thead>
                      <tr>
                        <th style={tableHeaderStyle}>Page ID</th>
                        <th style={tableHeaderStyle}>Model / Tensor Dim</th>
                        <th style={tableHeaderStyle}>Embedding Sample</th>
                        <th style={tableHeaderStyle}>Embeddings Count</th>
                        <th style={tableHeaderStyle}>Entry Matches</th>
                      </tr>
                    </thead>
                    <tbody>
                      {articleData()?.items?.length > 0 ? (
                        <For each={articleData().items}>
                          {(article) => (
                            <tr>
                              <td style={{ ...tableCellStyle, "color": "#0284c7", "font-weight": "bold" }}>{article.page_id}</td>
                              <td style={tableCellStyle}>{article.embedding_tensor.model} ({article.embedding_tensor.dim}d)</td>
                              <td style={{ ...tableCellStyle, "max-width": "220px", "overflow": "hidden", "text-overflow": "ellipsis", "white-space": "nowrap" }}>
                                {article.embedding_tensor.sample?.length ? `[${article.embedding_tensor.sample.map(n => n.toFixed(2)).join(", ")}...]` : "none"}
                              </td>
                              <td style={{ ...tableCellStyle, "color": "#059669", "font-weight": "bold" }}>{article.embedding_count}</td>
                              <td style={{ ...tableCellStyle, "max-width": "250px", "overflow": "hidden", "text-overflow": "ellipsis", "white-space": "nowrap", "color": "#0284c7" }}>
                                {article.entry_matches?.join(" | ") || "None"}
                              </td>
                            </tr>
                          )}
                        </For>
                      ) : (
                        <tr>
                          <td colSpan="5" style={{ ...tableCellStyle, "text-align": "center", "padding": "20px", "color": "#64748b" }}>No articles found matching criteria.</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>

                <div style={{ "display": "flex", "justify-content": "space-between", "align-items": "center", "margin-top": "10px" }}>
                  <span style={{ "font-size": "0.875rem", "color": "#64748b" }}>
                    Page {articleData()?.page || 1} of {articleData()?.pages || 1} (Total: {articleData()?.total || 0})
                  </span>
                  <div style={{ "display": "flex", "gap": "8px" }}>
                    <button 
                      onClick={() => setCurrentPage(p => Math.max(1, p - 1))} 
                      disabled={currentPage() <= 1}
                      style={{ "padding": "8px 16px", "background-color": "#e2e8f0", "color": "#0f172a", "border-radius": "6px", "border": "none", "cursor": currentPage() <= 1 ? "not-allowed" : "pointer", "opacity": currentPage() <= 1 ? 0.5 : 1 }}
                    >
                      Previous
                    </button>
                    <button 
                      onClick={() => setCurrentPage(p => Math.min((articleData()?.pages || 1), p + 1))} 
                      disabled={currentPage() >= (articleData()?.pages || 1)}
                      style={{ "padding": "8px 16px", "background-color": "#e2e8f0", "color": "#0f172a", "border-radius": "6px", "border": "none", "cursor": currentPage() >= (articleData()?.pages || 1) ? "not-allowed" : "pointer", "opacity": currentPage() >= (articleData()?.pages || 1) ? 0.5 : 1 }}
                    >
                      Next
                    </button>
                  </div>
                </div>
              </div>
            </Match>

            <Match when={activeTab() === "ingestion"}>
              <div style={{ "display": "flex", "flex-direction": "column", "gap": "20px", "width": "100%" }}>
                <div style={{ "display": "flex", "justify-content": "space-between", "align-items": "center", "flex-wrap": "wrap", "gap": "16px" }}>
                  <h3 style={{ "font-size": "1.2rem", "font-weight": "700", "color": "#059669", "margin": "0" }}>Wikipedia Ingestion & Chronological Metadata Feed</h3>
                  <div style={{ "display": "flex", "gap": "12px", "align-items": "center" }}>
                    <input 
                      type="number" 
                      min="1" 
                      max="100" 
                      value={articleCount()} 
                      onInput={(e) => setArticleCount(parseInt(e.target.value) || 5)}
                      style={{ ...inputStyle, "width": "90px" }}
                    />
                    <button onClick={(e) => handleDownloadSubmit(e)} style={{ "padding": "10px 20px", "background-color": "#2563eb", "color": "#ffffff", "border-radius": "6px", "font-weight": "600", "font-size": "0.95rem", "border": "none", "cursor": "pointer" }}>
                      Fetch Wikipedia
                    </button>
                  </div>
                </div>
                <div style={{ "font-size": "0.9rem", "color": "#0284c7", "font-weight": "600" }}>
                  Daemon Status: {downloadStatus()}
                </div>

                <div style={{ "background-color": "#f8fafc", "border": "1px solid #cbd5e1", "border-radius": "8px", "padding": "16px", "max-height": "350px", "overflow-y": "auto", "display": "flex", "flex-direction": "column", "gap": "10px" }}>
                  <div style={{ "font-size": "0.85rem", "font-weight": "700", "color": "#475569" }}>Chronological Article Metadata Scroll:</div>
                  {articleData()?.items?.length > 0 ? (
                    <For each={articleData().items}>
                      {(article) => (
                        <div style={{ "background": "#ffffff", "padding": "12px", "border-radius": "6px", "border": "1px solid #cbd5e1", "display": "flex", "flex-direction": "column", "gap": "4px" }}>
                          <div style={{ "display": "flex", "justify-content": "space-between", "font-size": "0.85rem" }}>
                            <strong style={{ "color": "#0284c7" }}>Page ID: {article.page_id}</strong>
                            <span style={{ "color": "#059669", "font-weight": "bold" }}>Embeddings: {article.embedding_count}</span>
                          </div>
                          <div style={{ "font-size": "0.8rem", "color": "#64748b" }}>Model: {article.embedding_tensor.model} ({article.embedding_tensor.dim}d)</div>
                          <div style={{ "font-family": "monospace", "font-size": "0.8rem", "color": "#334155", "overflow": "hidden", "text-overflow": "ellipsis", "white-space": "nowrap" }}>
                            {article.snippet}
                          </div>
                        </div>
                      )}
                    </For>
                  ) : (
                    <div style={{ "font-size": "0.85rem", "color": "#64748b" }}>No articles ingested yet. Use the button above to fetch!</div>
                  )}
                </div>
              </div>
            </Match>
          </Switch>
        </div>

        <div style={{ "display": "grid", "grid-template-columns": "repeat(auto-fit, minmax(450px, 1fr))", "gap": "32px", "width": "100%" }}>
          
          <div style={cardStyle}>
            <div style={{ "display": "flex", "justify-content": "space-between", "align-items": "center", "width": "100%" }}>
              <div style={{ "display": "flex", "align-items": "center", "gap": "12px" }}>
                <h2 style={{ "font-size": "1.5rem", "font-weight": "700", "color": "#0f172a", "margin": "0" }}>Model Training Daemon</h2>
                <span style={{ "font-size": "0.75rem", "padding": "4px 10px", "border-radius": "9999px", "background-color": isConverging() ? "#d1fae5" : "#fee2e2", "color": isConverging() ? "#065f46" : "#991b1b", "border": `1px solid ${isConverging() ? "#10b981" : "#ef4444"}`, "font-weight": "600" }}>
                  {isConverging() ? "● Converging Active" : "○ Adjusting..."}
                </span>
              </div>
              <button onClick={toggleTraining} style={{ "padding": "12px 24px", "border-radius": "6px", "font-weight": "600", "font-size": "1rem", "cursor": "pointer", "border": "none", "color": "#ffffff", "background-color": isTraining() ? "#e11d48" : "#059669" }}>
                {isTraining() ? "Pause" : "Start"}
              </button>
            </div>

            <div style={{ "display": "grid", "grid-template-columns": "repeat(3, minmax(0, 1fr))", "gap": "16px", "width": "100%" }}>
              <div style={{ "background-color": "#f8fafc", "padding": "16px", "border-radius": "6px", "border": "1px solid #cbd5e1" }}>
                <div style={{ "font-size": "0.875rem", "color": "#64748b" }}>Epoch</div>
                <div style={{ "font-size": "1.75rem", "font-weight": "800", "color": "#0f172a", "margin-top": "6px" }}>{epoch()}</div>
              </div>
              <div style={{ "background-color": "#f8fafc", "padding": "16px", "border-radius": "6px", "border": "1px solid #cbd5e1" }}>
                <div style={{ "font-size": "0.875rem", "color": "#64748b" }}>Samples</div>
                <div style={{ "font-size": "1.75rem", "font-weight": "800", "color": "#0f172a", "margin-top": "6px" }}>{totalSamples()}</div>
              </div>
              <div style={{ "background-color": "#f8fafc", "padding": "16px", "border-radius": "6px", "border": "1px solid #cbd5e1" }}>
                <div style={{ "font-size": "0.875rem", "color": "#64748b" }}>Gain</div>
                <div style={{ "font-size": "1.75rem", "font-weight": "800", "color": "#059669", "margin-top": "6px" }}>{fidelityGain().toFixed(1)}%</div>
              </div>
            </div>

            <div style={{ "display": "flex", "flex-direction": "column", "gap": "14px", "background-color": "#f8fafc", "padding": "16px", "border-radius": "8px", "border": "1px solid #cbd5e1", "width": "100%", "box-sizing": "border-box" }}>
              <div style={{ "display": "flex", "flex-direction": "column", "gap": "6px" }}>
                <div style={{ "display": "flex", "justify-content": "space-between", "font-size": "0.875rem", "color": "#64748b" }}>
                  <span>Full Loss (Δ: {lossDelta >= 0 ? `-${Math.abs(lossDelta).toFixed(4)}` : `+${Math.abs(lossDelta).toFixed(4)}`})</span>
                  <span style={{ "color": "#059669", "font-weight": "700" }}>{lossFull().toFixed(4)}</span>
                </div>
                <div style={{ "width": "100%", "background-color": "#e2e8f0", "border-radius": "9999px", "height": "8px", "border": "1px solid #cbd5e1", "overflow": "hidden" }}>
                  <div style={{ "background-color": "#10b981", "height": "100%", "width": `${Math.min(100, lossFull() * 100)}%` }}></div>
                </div>
              </div>
              <div style={{ "display": "flex", "flex-direction": "column", "gap": "6px" }}>
                <div style={{ "display": "flex", "justify-content": "space-between", "font-size": "0.875rem", "color": "#64748b" }}>
                  <span>Half Loss</span><span style={{ "color": "#0284c7", "font-weight": "700" }}>{lossHalf().toFixed(4)}</span>
                </div>
                <div style={{ "width": "100%", "background-color": "#e2e8f0", "border-radius": "9999px", "height": "8px", "border": "1px solid #cbd5e1", "overflow": "hidden" }}>
                  <div style={{ "background-color": "#0284c7", "height": "100%", "width": `${Math.min(100, lossHalf() * 100)}%` }}></div>
                </div>
              </div>
            </div>

            <div style={{ "background-color": "#f8fafc", "padding": "14px", "border-radius": "6px", "border": "1px solid #cbd5e1", "display": "flex", "flex-direction": "column", "gap": "8px" }}>
              <div style={{ "font-size": "0.85rem", "font-weight": "700", "color": "#0284c7" }}>Live Training Sample Activity (Click to Infeers Seed)</div>
              <div style={{ "display": "flex", "flex-direction": "column", "gap": "4px", "max-height": "140px", "overflow-y": "auto" }}>
                <For each={sampleLogs()}>
                  {(log) => (
                    <div 
                      onClick={() => runInference(log.tokenIds)}
                      style={{ "display": "flex", "justify-content": "space-between", "align-items": "center", "font-size": "0.8rem", "font-family": "monospace", "color": "#334155", "background": "#ffffff", "padding": "6px 8px", "border-radius": "4px", "border": "1px solid #cbd5e1", "cursor": "pointer" }}
                      title="Click to run inference using this trained sample sub-window!"
                    >
                      <span style={{ "overflow": "hidden", "text-overflow": "ellipsis", "white-space": "nowrap", "max-width": "70%" }}>[{log.docId}] {log.span}</span>
                      <span style={{ "color": "#059669", "flex-shrink": "0" }}>loss: {log.lossFull}</span>
                    </div>
                  )}
                </For>
              </div>
            </div>

          </div>

          <div style={cardStyle}>
            <div style={{ "display": "flex", "justify-content": "space-between", "align-items": "center", "width": "100%" }}>
              <h2 style={{ "font-size": "1.5rem", "font-weight": "700", "color": "#0f172a", "margin": "0" }}>Checkpoint Management</h2>
              <div style={{ "font-size": "0.875rem", "color": "#64748b" }}>Auto-Save: {lastAutoSave()}</div>
            </div>
            <div style={{ "display": "flex", "flex-direction": "column", "gap": "16px", "width": "100%" }}>
              <div style={{ "display": "flex", "gap": "12px", "width": "100%" }}>
                <select value={selectedCheckpoint()} onChange={(e) => setSelectedCheckpoint(e.target.value)} style={{ ...inputStyle, "flex": "1" }}>
                  <option value="">-- Select Checkpoint --</option>
                  <For each={checkpoints()}>{(c) => <option value={c.id}>{c.name} (Ep {c.epoch})</option>}</For>
                </select>
                <button onClick={triggerLoadCheckpoint} style={{ "padding": "12px 24px", "background-color": "#475569", "color": "#ffffff", "border-radius": "6px", "font-size": "1rem", "border": "none", "cursor": "pointer", "font-weight": "600" }}>Load</button>
              </div>
              <button onClick={triggerAutoSave} disabled={isSaving()} style={{ "width": "100%", "padding": "12px 24px", "background-color": "#059669", "color": "#ffffff", "border-radius": "6px", "font-size": "1rem", "border": "none", "cursor": "pointer", "font-weight": "600" }}>
                {isSaving() ? "Saving Checkpoint..." : "Save Checkpoint Now"}
              </button>
            </div>
          </div>

        </div>

        <div style={cardStyle}>
          <div style={{ "display": "flex", "justify-content": "space-between", "align-items": "center", "flex-wrap": "wrap", "gap": "16px", "width": "100%" }}>
            <div>
              <h2 style={{ "font-size": "1.5rem", "font-weight": "700", "color": "#0f172a", "margin": "0" }}>Inference & nanoGPT Decoding Sandbox</h2>
              <p style={{ "font-size": "0.95rem", "color": "#64748b", "margin": "4px 0 0 0" }}>Test generation using Temperature, Top-K, and Top-P (Nucleus) sampling</p>
            </div>
            <button onClick={() => runInference(null)} disabled={isInferring()} style={{ "padding": "12px 28px", "background-color": "#2563eb", "color": "#ffffff", "border-radius": "6px", "font-weight": "600", "font-size": "1rem", "border": "none", "cursor": "pointer" }}>
              {isInferring() ? "Sampling..." : "Run Inference"}
            </button>
          </div>

          <div style={{ "display": "grid", "grid-template-columns": "repeat(auto-fit, minmax(200px, 1fr))", "gap": "16px", "width": "100%" }}>
            <div style={{ "background-color": "#f8fafc", "padding": "12px", "border-radius": "6px", "border": "1px solid #cbd5e1", "display": "flex", "flex-direction": "column", "gap": "6px" }}>
              <label style={{ "font-size": "0.85rem", "color": "#64748b" }}>Temperature: <strong style={{ "color": "#0284c7" }}>{temperature()}</strong></label>
              <input type="range" min="0.1" max="2.0" step="0.1" value={temperature()} onInput={(e) => setTemperature(parseFloat(e.target.value))} style={{ "width": "100%" }} />
            </div>
            <div style={{ "background-color": "#f8fafc", "padding": "12px", "border-radius": "6px", "border": "1px solid #cbd5e1", "display": "flex", "flex-direction": "column", "gap": "6px" }}>
              <label style={{ "font-size": "0.85rem", "color": "#64748b" }}>Top-K: <strong style={{ "color": "#0284c7" }}>{topK()}</strong></label>
              <input type="range" min="0" max="100" step="5" value={topK()} onInput={(e) => setTopK(parseInt(e.target.value))} style={{ "width": "100%" }} />
            </div>
            <div style={{ "background-color": "#f8fafc", "padding": "12px", "border-radius": "6px", "border": "1px solid #cbd5e1", "display": "flex", "flex-direction": "column", "gap": "10px" }}>
              <label style={{ "font-size": "0.85rem", "color": "#64748b" }}>Top-P (Nucleus): <strong style={{ "color": "#0284c7" }}>{topP()}</strong></label>
              <input type="range" min="0.1" max="1.0" step="0.05" value={topP()} onInput={(e) => setTopP(parseFloat(e.target.value))} style={{ "width": "100%" }} />
            </div>
            <div style={{ "background-color": "#f8fafc", "padding": "12px", "border-radius": "6px", "border": "1px solid #cbd5e1", "display": "flex", "flex-direction": "column", "gap": "6px" }}>
              <label style={{ "font-size": "0.85rem", "color": "#64748b" }}>Target Tokens: <strong style={{ "color": "#0284c7" }}>{targetTokens()}</strong></label>
              <input type="number" min="4" max="64" value={targetTokens()} onInput={(e) => setTargetTokens(parseInt(e.target.value) || 16)} style={{ ...inputStyle, "padding": "6px 10px", "font-size": "0.9rem" }} />
            </div>
          </div>

          {diffusionResult() && (
            <div style={{ "display": "flex", "flex-direction": "column", "gap": "16px", "background-color": "#f8fafc", "padding": "20px", "border-radius": "8px", "border": "1px solid #cbd5e1" }}>
              <div style={{ "display": "flex", "justify-content": "space-between", "font-size": "0.9rem", "color": "#64748b" }}>
                <span>Seed Document: <strong style={{ "color": "#0f172a" }}>{diffusionResult().docId}</strong></span>
                <span>Seed Prompt: <strong style={{ "color": "#059669" }}>{diffusionResult().seedDecoded}</strong></span>
              </div>
              <div style={{ "font-family": "monospace", "font-size": "1rem", "color": "#0f172a", "background": "#ffffff", "padding": "16px", "border-radius": "6px", "border": "1px solid #cbd5e1", "word-break": "break-all" }}>
                <strong style={{ "color": "#0284c7" }}>Decoded Output:</strong> {diffusionResult().trajectoryFull?.[diffusionResult().trajectoryFull.length - 1]?.pureCompletion}
              </div>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}