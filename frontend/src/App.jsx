import React, { useState } from "react";

const API_BASE = "http://localhost:8000"; // FastAPI backend

const TABS = ["Summary", "Entities", "QA", "Combined", "Chat", "Risk"];

// Simple loading spinner component
const Spinner = () => (
  <div className="spinner" aria-label="Loading">
    <div className="spinner-circle" />
  </div>
);


function App() {
  const [activeTab, setActiveTab] = useState("Summary");
  const [text, setText] = useState("");
  const [question, setQuestion] = useState("");

  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState("");
  const [entities, setEntities] = useState([]);
  const [qaResult, setQaResult] = useState(null);
  const [error, setError] = useState("");
  // Summary mode: local T5, Groq, or RAG
  const [summaryMode, setSummaryMode] = useState("local"); // "local" | "groq" | "rag"

  // QA mode: "extractive" (local RoBERTa) or "generative" (Groq Llama3 via /qa_gen)
  const [qaMode, setQaMode] = useState("extractive");

  const [chatMessages, setChatMessages] = useState([]); // [{role, content}]
  const [chatInput, setChatInput] = useState("");

  const [riskResult, setRiskResult] = useState(null);
  const [sectionRisks, setSectionRisks] = useState([]);
  const [summaryEngine, setSummaryEngine] = useState("t5"); // "t5" | "bart"


  const resetOutputs = () => {
    setSummary("");
    setEntities([]);
    setQaResult(null);
    setError("");
  };

  const handlePdfUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    resetOutputs();
    setError("");
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_BASE}/extract_text`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Error extracting text from PDF.");
      }

      // Put extracted text into the main textarea
      setText(data.text || "");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      // reset input so re-uploading same file works
      event.target.value = "";
    }
  };

  const handleSummarizeLocal = async () => {
    if (!text.trim()) {
      setError("Please enter some text to summarize.");
      return;
    }
    resetOutputs();
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          max_new_tokens: 256,
          engine: summaryEngine, // "t5" or "bart"
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Error from API");
      setSummary(data.summary);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSummarizeGroq = async () => {
    if (!text.trim()) {
      setError("Please enter some text to summarize.");
      return;
    }
    resetOutputs();
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/summarize_groq`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, max_new_tokens: 512 }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Error from Groq summary API");
      setSummary(data.summary);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSummarizeRag = async () => {
    if (!text.trim()) {
      setError("Please enter some text to summarize.");
      return;
    }
    resetOutputs();
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/summarize_rag`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          max_new_tokens: 512,
          top_k: 5,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Error from RAG summary API");
      setSummary(data.summary);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  
  const handleRisk = async () => {
    if (!text.trim()) {
      setError("Please enter or paste some text to analyze risk.");
      return;
    }
    setError("");
    setLoading(true);
    setRiskResult(null);

    try {
      const res = await fetch(`${API_BASE}/risk`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Error from risk API");

      setRiskResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRiskSections = async () => {
    if (!text.trim()) {
      setError("Please enter or paste some text to analyze section-wise risk.");
      return;
    }
    setError("");
    setLoading(true);
    setSectionRisks([]);

    try {
      const res = await fetch(`${API_BASE}/risk_sections`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Error from section risk API.");

      setSectionRisks(data.sections || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleEntities = async () => {
    if (!text.trim()) {
      setError("Please enter some text to analyze entities.");
      return;
    }
    resetOutputs();
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/ner`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Error from API");
      setEntities(data.entities || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  // Extractive QA (local RoBERTa /qa)
  const handleQA = async () => {
    if (!text.trim()) {
      setError("Please enter context text.");
      return;
    }
    if (!question.trim()) {
      setError("Please enter a question.");
      return;
    }
    resetOutputs();
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/qa`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ context: text, question }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Error from API");
      setQaResult(data); // { answer, score, start, end }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  // Generative QA (Groq Llama3 /qa_gen)
  const handleQAGen = async () => {
    if (!text.trim()) {
      setError("Please enter context text.");
      return;
    }
    if (!question.trim()) {
      setError("Please enter a question.");
      return;
    }
    resetOutputs();
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/qa_gen`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          context: text,
          question,
          max_new_tokens: 128,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Error from API");
      // data = { answer }
      setQaResult({
        answer: data.answer,
        score: null,
        start: -1,
        end: -1,
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleQARag = async () => {
    if (!text.trim()) {
      setError("Please enter context text.");
      return;
    }
    if (!question.trim()) {
      setError("Please enter a question.");
      return;
    }
    resetOutputs();
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/qa_rag`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          context: text,
          question,
          top_k: 3,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Error from RAG QA API.");

      // data = { answer, retrieved_chunks: [...] }
      setQaResult({
        answer: data.answer,
        score: null,
        start: -1,
        end: -1,
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadSummary = () => {
    if (!summary) return;
    const blob = new Blob([summary], {
      type: "text/plain;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "summary.txt";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

const handleSendChat = async () => {
  const trimmed = chatInput.trim();
  if (!trimmed) return;

  if (!text.trim()) {
    setError("Please provide context text in the left panel before chatting.");
    return;
  }

  // 1) Add user's message to history immediately
  const newUserMessage = { role: "user", content: trimmed };
  const newHistory = [...chatMessages, newUserMessage];
  setChatMessages(newHistory);
  setChatInput("");
  setError("");
  setLoading(true);

  try {
    const res = await fetch(`${API_BASE}/chat_qa`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        context: text,
        messages: newHistory,
      }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Error from chat QA API.");
    }

    // 2) Add assistant's reply to chat history
    const assistantMessage = {
      role: "assistant",
      content: data.reply || "(no reply)",
    };
    setChatMessages((prev) => [...prev, assistantMessage]);
  } catch (e) {
    setError(e.message);
  } finally {
    setLoading(false);
  }
};

  const handleNewChat = () => {
    setChatMessages([]);
    setChatInput("");
    setError("");
  };

  const handleDownloadEntities = () => {
    if (!entities || entities.length === 0) return;
    const json = JSON.stringify(entities, null, 2);
    const blob = new Blob([json], {
      type: "application/json;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "entities.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError("Please enter text to analyze.");
      return;
    }
    resetOutputs();
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          question: question || null,
          max_new_tokens: 256,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Error from API");
      setSummary(data.summary);
      setEntities(data.entities || []);
      setQaResult(data.qa || null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const renderOutputs = () => (
    <div className="outputs">
      {error && <div className="error">⚠ {error}</div>}

      {/* Centered spinner when loading and no results yet */}
      {loading && !summary && entities.length === 0 && !qaResult && !error && (
        <div className="spinner-container">
          <Spinner />
          <p className="spinner-text">Thinking...</p>
        </div>
      )}

      {summary && (
        <div className="card">
          <h3>Summary</h3>
          <p>{summary}</p>
          <div className="card-actions">
            <button
              type="button"
              className="secondary-btn"
              onClick={handleDownloadSummary}
            >
              Download Summary (.txt)
            </button>
          </div>
        </div>
      )}

      {entities && entities.length > 0 && (
        <div className="card">
          <h3>Entities</h3>
          <div className="entities-list">
            {entities.map((ent, idx) => (
              <span key={idx} className="entity-chip">
                <strong>{ent.text}</strong> <span>[{ent.label}]</span>
              </span>
            ))}
          </div>
          <div className="card-actions">
            <button
              type="button"
              className="secondary-btn"
              onClick={handleDownloadEntities}
            >
              Download Entities (.json)
            </button>
          </div>
        </div>
      )}

      {qaResult && (
        <div className="card">
          <h3>QA Result</h3>
          <p>
            <strong>Answer:</strong> {qaResult.answer || "(no answer found)"}
          </p>
          {qaMode === "extractive" && qaResult.score !== null && (
            <p>
              <strong>Score:</strong>{" "}
              {qaResult.score ? qaResult.score.toFixed(3) : "N/A"}
            </p>
          )}
        </div>
      )}
    </div>
  );


  const renderTabContent = () => {
    switch (activeTab) {
      case "Summary":
        return (
          <>
            {/* Summary mode selector (local / groq / rag) */}
            <div className="qa-mode-toggle">
              <span style={{ fontSize: "0.8rem", color: "#9ca3af" }}>
                Summary Mode:
              </span>
              <div className="qa-mode-buttons">
                <button
                  type="button"
                  className={`mode-btn ${
                    summaryMode === "local" ? "mode-active" : ""
                  }`}
                  onClick={() => {
                    resetOutputs();
                    setSummaryMode("local");
                  }}
                >
                  Model (Local)
                </button>
                <button
                  type="button"
                  className={`mode-btn ${
                    summaryMode === "groq" ? "mode-active" : ""
                  }`}
                  onClick={() => {
                    resetOutputs();
                    setSummaryMode("groq");
                  }}
                >
                  Generative (Groq)
                </button>
                <button
                  type="button"
                  className={`mode-btn ${
                    summaryMode === "rag" ? "mode-active" : ""
                  }`}
                  onClick={() => {
                    resetOutputs();
                    setSummaryMode("rag");
                  }}
                >
                  RAG (Embed + Groq)
                </button>
              </div>
            </div>

            {/* T5 / BART selector shown only when using local model */}
            {summaryMode === "local" && (
              <div className="qa-mode-toggle" style={{ marginTop: "0.4rem" }}>
                <span style={{ fontSize: "0.8rem", color: "#9ca3af" }}>
                  Local Model:
                </span>
                <div className="qa-mode-buttons">
                  <button
                    type="button"
                    className={`mode-btn ${
                      summaryEngine === "t5" ? "mode-active" : ""
                    }`}
                    onClick={() => {
                      resetOutputs();
                      setSummaryEngine("t5");
                    }}
                  >
                    T5 (fine-tuned)
                  </button>
                  <button
                    type="button"
                    className={`mode-btn ${
                      summaryEngine === "bart" ? "mode-active" : ""
                    }`}
                    onClick={() => {
                      resetOutputs();
                      setSummaryEngine("bart");
                    }}
                  >
                    BART (fine-tuned)
                  </button>
                </div>
              </div>
            )}

            <button
              className="primary-btn"
              onClick={
                summaryMode === "local"
                  ? handleSummarizeLocal
                  : summaryMode === "groq"
                  ? handleSummarizeGroq
                  : handleSummarizeRag
              }
              disabled={loading}
            >
              {loading
                ? summaryMode === "local"
                  ? summaryEngine === "t5"
                    ? "Summarizing (T5)..."
                    : "Summarizing (BART)..."
                  : summaryMode === "groq"
                  ? "Summarizing (Groq LLM)..."
                  : "Summarizing (RAG + Groq)..."
                : summaryMode === "local"
                ? summaryEngine === "t5"
                  ? "Summarize (T5 fine-tuned)"
                  : "Summarize (BART fine-tuned)"
                : summaryMode === "groq"
                ? "Summarize (Groq LLM)"
                : "Summarize (RAG + Groq)"}
            </button>

            {renderOutputs()}
          </>
        );

      case "Entities":
        return (
          <>
            <button
              className="primary-btn"
              onClick={handleEntities}
              disabled={loading}
            >
              {loading ? "Extracting..." : "Extract Entities"}
            </button>
            {renderOutputs()}
          </>
        );

      case "QA":
        return (
          <>
            <div className="qa-mode-toggle">
              <span style={{ fontSize: "0.8rem", color: "#9ca3af" }}>
                QA Mode:
              </span>
              <div className="qa-mode-buttons">
                <button
                  type="button"
                  className={`mode-btn ${
                    qaMode === "extractive" ? "mode-active" : ""
                  }`}
                  onClick={() => {
                    resetOutputs();
                    setQaMode("extractive");
                  }}
                >
                  Extractive
                </button>
                <button
                  type="button"
                  className={`mode-btn ${
                    qaMode === "generative" ? "mode-active" : ""
                  }`}
                  onClick={() => {
                    resetOutputs();
                    setQaMode("generative");
                  }}
                >
                  Generative (Groq)
                </button>
                <button
                  type="button"
                  className={`mode-btn ${
                    qaMode === "rag" ? "mode-active" : ""
                  }`}
                  onClick={() => {
                    resetOutputs();
                    setQaMode("rag");
                  }}
                >
                  RAG (Embed + Groq)
                </button>
              </div>
            </div>

            <label className="label">
              Question
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder='e.g., What does the term "nonprofit organization" mean?'
              />
            </label>

            <button
              className="primary-btn"
              onClick={
                qaMode === "extractive"
                  ? handleQA
                  : qaMode === "generative"
                  ? handleQAGen
                  : handleQARag
              }
              disabled={loading}
            >
              {loading
                ? qaMode === "extractive"
                  ? "Answering (extractive)..."
                  : qaMode === "generative"
                  ? "Answering (generative)..."
                  : "Answering (RAG+Groq)..."
                : qaMode === "extractive"
                ? "Ask (Extractive QA)"
                : qaMode === "generative"
                ? "Ask (Generative QA via Groq)"
                : "Ask (RAG: Retrieve + Groq)"}
            </button>

            {renderOutputs()}
          </>
        );
        

      case "Combined":
        return (
          <>
            <label className="label">
              Question (optional)
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask your question about the text (optional)"
              />
            </label>
            <button
              className="primary-btn"
              onClick={handleAnalyze}
              disabled={loading}
            >
              {loading ? "Analyzing..." : "Analyze (Summary + NER + QA)"}
            </button>
            {renderOutputs()}
          </>
        );

      case "Chat":
        return (
          <>
            <div className="card" style={{ maxHeight: "380px", overflow: "hidden" }}>
              <div className="chat-header-row">
                <h3>Chat with Legal Assistant</h3>
                <button
                  type="button"
                  className="secondary-btn"
                  onClick={handleNewChat}
                >
                  New Chat
                </button>
              </div>
              <div className="chat-container">
                <div className="chat-messages">
                  {chatMessages.length === 0 && (
                    <div className="chat-empty">
                      Ask questions about the document on the left.
                    </div>
                  )}
                  {chatMessages.map((msg, idx) => (
                    <div
                      key={idx}
                      className={`chat-bubble ${
                        msg.role === "user"
                          ? "chat-bubble-user"
                          : "chat-bubble-assistant"
                      }`}
                    >
                      <div className="chat-role">
                        {msg.role === "user" ? "You" : "Assistant"}
                      </div>
                      <div className="chat-content">{msg.content}</div>
                    </div>
                  ))}
                  {loading && (
                    <div className="chat-bubble chat-bubble-assistant">
                      <div className="chat-role">Assistant</div>
                      <div className="chat-content">
                        <Spinner />{" "}
                        <span style={{ marginLeft: 6 }}>Thinking...</span>
                      </div>
                    </div>
                  )}
                </div>

                <div className="chat-input-row">
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Ask a follow-up question about the document..."
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        handleSendChat();
                      }
                    }}
                  />
                  <button
                    type="button"
                    className="primary-btn chat-send-btn"
                    onClick={handleSendChat}
                    disabled={loading}
                  >
                    Send
                  </button>
                </div>
              </div>
            </div>
          </>
        );

      case "Risk":
        return (
          <>
            <div className="risk-buttons-row">
              <button
                className="primary-btn"
                onClick={handleRisk}
                disabled={loading}
              >
                {loading ? "Analyzing risk..." : "Analyze Overall Risk"}
              </button>
              <button
                type="button"
                className="secondary-btn"
                onClick={handleRiskSections}
                disabled={loading}
              >
                {loading ? "Analyzing sections..." : "Analyze Section-wise Risk"}
              </button>
            </div>

            {riskResult && (
              <div className="card" style={{ marginTop: "0.75rem" }}>
                <h3>Overall Legal Risk</h3>
                <p>
                  <strong>Risk level:</strong> {riskResult.top_label}
                </p>
                <div style={{ marginTop: "0.5rem", fontSize: "0.8rem" }}>
                  {Object.entries(riskResult.scores).map(([label, score]) => (
                    <div key={label} style={{ marginBottom: "0.25rem" }}>
                      {label}: {(score * 100).toFixed(1)}%
                    </div>
                  ))}
                </div>
              </div>
            )}

            {sectionRisks && sectionRisks.length > 0 && (
              <div className="card" style={{ marginTop: "0.75rem" }}>
                <h3>Section-wise Risk Map</h3>
                <div className="section-risk-list">
                  {sectionRisks.map((sec, idx) => (
                    <div key={idx} className="section-risk-item">
                      <div className="section-risk-header">
                        <span className="section-title">{sec.title}</span>
                        <span
                          className={`risk-badge risk-${
                            (sec.top_label || "").toLowerCase().split(" ")[0]
                          }`}
                        >
                          {sec.top_label}
                        </span>
                      </div>
                      <p className="section-text-preview">
                        {sec.text.length > 300
                          ? sec.text.slice(0, 300) + "..."
                          : sec.text}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        );



      default:
        return null;
    }
  };

  return (
    <div className="app">
      <header>
        <h1>Legal Document Assistant</h1>
        <p className="subtitle">
          Summarization • Entity Extraction • Question Answering • Legal Document Assistant Bot • Risk Analysis
        </p>
      </header>

      <div className="layout">
        <div className="left-panel">
          <div className="upload-row">
            <label className="label" style={{ flex: 1 }}>
              Input Text
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste a legal/policy document or long text here, or upload a PDF below..."
                rows={16}
              />
            </label>
          </div>

          <div className="upload-row" style={{ marginTop: "0.75rem" }}>
            <span style={{ fontSize: "0.8rem", color: "#9ca3af" }}>
              Or upload a PDF document:
            </span>
            <input
              type="file"
              accept="application/pdf"
              onChange={handlePdfUpload}
              className="file-input"
            />
          </div>
        </div>

        <div className="right-panel">
          <div className="tabs">
            {TABS.map((tab) => (
              <button
                key={tab}
                className={`tab-btn ${
                  activeTab === tab ? "tab-active" : ""
                }`}
                onClick={() => {
                  resetOutputs();
                  setActiveTab(tab);
                }}
              >
                {tab}
              </button>
            ))}
          </div>

          <div className="tab-content">{renderTabContent()}</div>
        </div>
      </div>
    </div>
  );
}

export default App;
