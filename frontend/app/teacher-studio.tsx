"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ComponentType } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function apiFetch(
  input: RequestInfo | URL,
  init?: RequestInit,
  retries = 12,
): Promise<Response> {
  let lastError: unknown;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const response = await fetch(input, init);

      // Backend is alive. Return the response normally.
      return response;
    } catch (error) {
      lastError = error;
      if (init?.signal?.aborted || (error instanceof DOMException && error.name === "AbortError")) throw error;

      if (attempt === retries) {
        throw lastError;
      }

      // Backend may still be initializing. Wait progressively,
      // but never block the UI itself.
      await new Promise<void>((resolve, reject) => {
        const signal = init?.signal;
        const timeout = window.setTimeout(() => {
          signal?.removeEventListener("abort", abort);
          resolve();
        }, Math.min(1000 + attempt * 500, 3000));
        const abort = () => {
          window.clearTimeout(timeout);
          reject(new DOMException("The operation was cancelled.", "AbortError"));
        };
        if (signal?.aborted) abort();
        else signal?.addEventListener("abort", abort, { once: true });
      });
    }
  }

  throw lastError;
}

function createOperation() {
  return { id: crypto.randomUUID(), controller: new AbortController() };
}

function cancelBackendOperation(operation: { id: string; controller: AbortController } | null, token: string) {
  if (!operation) return;
  operation.controller.abort();
  void fetch(`${API}/operations/cancel?operation_id=${encodeURIComponent(operation.id)}`, {
    method: "POST",
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  }).catch(() => {
    // The local operation may have already completed; closing the UI is still immediate.
  });
}

function useElapsed(active: boolean) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!active) {
      setElapsed(0);
      return;
    }

    const start = Date.now();
    const id = window.setInterval(() => {
      setElapsed(Math.floor((Date.now() - start) / 1000));
    }, 250);

    return () => window.clearInterval(id);
  }, [active]);

  return elapsed;
}

type Bloom = { level: string; confidence: number };
type Moderation = {
  bloom_level: string;
  target_higher_level: string;
  reason: string;
  higher_level_rewrite: string;
};

type LoadingProps = {
  label: string;
  hint?: string;
  success?: boolean;
  successMessage?: string;
  finalElapsed?: number;
  showGame?: boolean;
  onCancel?: () => void;
};

function TeacherStudio({ token, LoadingComponent }: { token: string; LoadingComponent: ComponentType<LoadingProps> }) {
  const [q, setQ] = useState("");
  const [bloom, setBloom] = useState<Bloom | null>(null);
  const [mod, setMod] = useState<Moderation | null>(null);
  const [linguisticMod, setLinguisticMod] = useState<{ original: string; improved: string } | null>(null);
  const [targetRewrite, setTargetRewrite] = useState<{ original: string; rewritten: string; target_level: string } | null>(null);
  const [targetLevel, setTargetLevel] = useState("Analyze");
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [dropdownPosition, setDropdownPosition] = useState({ top: 0, left: 0 });
  const dropdownRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setDropdownOpen(false);
      }
    }

    if (dropdownOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [dropdownOpen]);

  // Calculate dropdown position when opening
  const handleDropdownToggle = () => {
    if (!dropdownOpen && triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      const dropdownHeight = 280; // Approximate height of dropdown menu
      const viewportHeight = window.innerHeight;
      const spaceBelow = viewportHeight - rect.bottom;
      
      let topPosition;
      if (spaceBelow < dropdownHeight + 20) {
        // Not enough space below, position above the button
        topPosition = rect.top - dropdownHeight - 8;
      } else {
        // Enough space below, position below the button
        topPosition = rect.bottom + 8;
      }
      
      setDropdownPosition({
        top: topPosition,
        left: rect.left
      });
    }
    setDropdownOpen(!dropdownOpen);
  };
  const [answer, setAnswer] = useState("");
  const [summary, setSummary] = useState("");
  const [busy, setBusy] = useState("");
  const operationController = useRef<ReturnType<typeof createOperation> | null>(null);
  const cancelOperation = useCallback(() => {
    const active = operationController.current;
    operationController.current = null;
    cancelBackendOperation(active, token);
    setBusy("");
    setSuccess({ show: false, message: "", elapsed: 0 });
  }, [token]);
  const [error, setError] = useState("");
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [success, setSuccess] = useState<{ show: boolean; message: string; elapsed: number }>({ show: false, message: "", elapsed: 0 });
  const operationElapsed = useElapsed(!!busy);
  const [classifierLoaded, setClassifierLoaded] = useState(false);
  const [classifierLoadAttempt, setClassifierLoadAttempt] = useState(0);

  async function loadBloomClassifier() {
    // Load silently in the background; do not replace or block the teacher UI.
    try {
      const response = await apiFetch(`${API}/models/load-bloom`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
      const result = await response.json();
      if (!response.ok) throw Error(result.detail || "The teacher Bloom classifier could not be loaded.");
      setClassifierLoaded(true);
    } catch (e) {
      // Surface only failures so the teacher can retry; successful loading is silent.
      setError(String(e).replace("Error: ", ""));
    }
  }

  // Start loading only after the authenticated teacher workspace has mounted.
  useEffect(() => {
    if (!classifierLoaded) void loadBloomClassifier();
    // Initial mount and explicit retry are the only triggers.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [classifierLoadAttempt, token]);

  // Clear previous output when input changes
  const handleQuestionChange = (newQuestion: string) => {
    setQ(newQuestion);
    // Clear all previous outputs when question changes
    setBloom(null);
    setMod(null);
    setLinguisticMod(null);
    setTargetRewrite(null);
    setAnswer("");
    setSummary("");
    setError("");
    setSuccess({ show: false, message: "", elapsed: 0 });
  };

  // Clear previous output when target level changes
  const handleTargetLevelChange = (newLevel: string) => {
    setTargetLevel(newLevel);
    setDropdownOpen(false);
    // Clear only target rewrite output when target level changes
    setTargetRewrite(null);
    setError("");
    setSuccess({ show: false, message: "", elapsed: 0 });
    // Also clear any previous validation-related states
    setBloom(null);
    setMod(null);
    setLinguisticMod(null);
    setAnswer("");
    setSummary("");
  };

  // Handle target level selection and auto-trigger rewrite
  const handleTargetLevelSelect = (level: string) => {
    handleTargetLevelChange(level);
    // Auto-trigger rewrite after selection
    if (q.trim()) {
      run("target-rewrite", level);
    }
  };

  async function req(path: string, body: object) {
    const active = operationController.current;
    const r = await apiFetch(`${API}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
        ...(active ? { "X-Operation-ID": active.id } : {}),
      },
      body: JSON.stringify(body),
      signal: active?.controller.signal,
    });
    const d = await r.json();
    if (!r.ok) throw Error(d.detail || "The local service could not complete this request.");
    return d;
  }

  async function run(action: "classify" | "moderate" | "linguistic" | "target-rewrite" | "qa" | "summary", explicitTargetLevel?: string) {
    // Target rewrites no longer use classifier re-scoring, so they can run while the classifier loads.
    if (!q.trim() || (!classifierLoaded && action !== "summary" && action !== "target-rewrite")) return;

    // Clear previous outputs when starting a new task
    setBloom(null);
    setMod(null);
    setLinguisticMod(null);
    setTargetRewrite(null);
    setAnswer("");
    setSummary("");
    setError("");
    setSuccess({ show: false, message: "", elapsed: 0 });

    const labels = {
      classify: "Classifying the question",
      moderate: "Preparing moderation guidance",
      linguistic: "Improving question quality",
      "target-rewrite": "Rewriting to target level",
      qa: "Loading the local model and writing a reply",
      summary: "Preparing a local summary",
    } as const;

    const successMessages = {
      classify: "Bloom level classified",
      moderate: "Moderated and rewritten",
      linguistic: "Question improved",
      "target-rewrite": "Question rewritten",
      qa: "Answered the question",
      summary: "Summarized the content",
    } as const;

    const operation = createOperation();
    operationController.current = operation;
    setBusy(labels[action]);

    try {
      if (action === "classify") {
        const d = await req("/teacher/exam/classify", { question: q });
        setBloom({ level: d.level, confidence: d.confidence });
        setSuccess({ show: true, message: successMessages.classify, elapsed: operationElapsed });
      }
      if (action === "moderate") {
        const d = await req("/teacher/exam/moderate", { question: q });
        setBloom({ level: d.bloom.level, confidence: d.bloom.confidence });
        setMod(d.moderation);
        setSuccess({ show: true, message: successMessages.moderate, elapsed: operationElapsed });
      }
      if (action === "linguistic") {
        const d = await req("/teacher/exam/linguistic-moderate", { question: q });
        setLinguisticMod({ original: d.original, improved: d.improved });
        setSuccess({ show: true, message: successMessages.linguistic, elapsed: operationElapsed });
      }
      if (action === "target-rewrite") {
        // Use explicit target level if provided (from click), otherwise use state
        const effectiveTargetLevel = explicitTargetLevel || targetLevel;
        const d = await req("/teacher/exam/target-rewrite", { question: q, target_level: effectiveTargetLevel });
        
        if (!d.rewritten) {
          setError(d.error || "Could not generate a rewrite.");
          setTargetRewrite(null);
        } else {
          setTargetRewrite({
            original: d.original,
            rewritten: d.rewritten,
            target_level: d.target_level,
          });
          setSuccess({ show: true, message: successMessages["target-rewrite"], elapsed: operationElapsed });
        }
      }
      if (action === "qa") {
        const hasContent = !!text.trim() || !!file;
        const scope = hasContent ? "public" : "public";
        const d = await req("/qa", { question: q, scope, top_k: 4 });
        setAnswer(d.answer);
        setSuccess({ show: true, message: successMessages.qa, elapsed: operationElapsed });
      }
      if (action === "summary") {
        const hasContent = !!text.trim() || !!file;
        const scope = hasContent ? "public" : "public";
        const d = await req("/summarize", { question: q, scope, top_k: 4 });
        setSummary(d.answer);
        setSuccess({ show: true, message: successMessages.summary, elapsed: operationElapsed });
      }
    } catch (e) {
      if (operationController.current === operation && !(e instanceof DOMException && e.name === "AbortError")) setError(String(e).replace("Error: ", ""));
    } finally {
      const stillCurrent = operationController.current === operation;
      if (stillCurrent) operationController.current = null;
      const finalElapsed = operationElapsed;
      if (stillCurrent) setBusy("");
      if (stillCurrent && successMessages[action] && !operation.controller.signal.aborted) {
        setSuccess(prev => ({ ...prev, elapsed: finalElapsed }));
        setTimeout(() => setSuccess({ show: false, message: "", elapsed: 0 }), 2000);
      }
    }
  }

  async function index() {
    if (!text.trim()) return;
    const operation = createOperation();
    operationController.current = operation;
    setBusy("Indexing your material");
    setError("");
    setSuccess({ show: false, message: "", elapsed: 0 });
    try {
      await req("/documents/index", {
        text,
        name: "study-notes",
        scope: "public",
        content_type: "study_material",
      });
      setSuccess({ show: true, message: "Indexing complete", elapsed: operationElapsed });
    } catch (e) {
      if (operationController.current === operation && !(e instanceof DOMException && e.name === "AbortError")) setError(String(e).replace("Error: ", ""));
    } finally {
      const stillCurrent = operationController.current === operation;
      if (stillCurrent) operationController.current = null;
      const finalElapsed = operationElapsed;
      if (stillCurrent) {
        setBusy("");
        if (!operation.controller.signal.aborted) setSuccess(prev => ({ ...prev, elapsed: finalElapsed }));
        setTimeout(() => setSuccess({ show: false, message: "", elapsed: 0 }), 2000);
      }
    }
  }

  async function upload() {
    if (!file) return;
    const operation = createOperation();
    operationController.current = operation;
    setBusy("Uploading and indexing your file");
    setError("");
    setSuccess({ show: false, message: "", elapsed: 0 });
    try {
      const f = new FormData();
      f.set("file", file);
      f.set("scope", "public");
      f.set("content_type", "study_material");
      const r = await apiFetch(`${API}/documents/upload`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}`, "X-Operation-ID": operation.id },
        body: f,
        signal: operation.controller.signal,
      });
      if (!r.ok) throw Error("Upload failed");
      setSuccess({ show: true, message: "Indexing complete", elapsed: operationElapsed });
    } catch (e) {
      if (operationController.current === operation && !(e instanceof DOMException && e.name === "AbortError")) setError(String(e).replace("Error: ", ""));
    } finally {
      const stillCurrent = operationController.current === operation;
      if (stillCurrent) operationController.current = null;
      const finalElapsed = operationElapsed;
      if (stillCurrent) {
        setBusy("");
        if (!operation.controller.signal.aborted) setSuccess(prev => ({ ...prev, elapsed: finalElapsed }));
        setTimeout(() => setSuccess({ show: false, message: "", elapsed: 0 }), 2000);
      }
    }
  }

  const level = mod?.bloom_level || bloom?.level;
  const loadingHint =
    busy === "Classifying the question" || busy === "Preparing moderation guidance"
      ? "Reviewing the question against local Bloom guidance."
      : "Your local model is preparing a grounded response.";
  const showSuccess = success.show && !busy;

  return (
    <main className="page-wrap fade-in teacher-workspace">
      <section className="teacher-hero glass-card rounded-[30px] p-5 sm:p-8">
        <div className="teacher-hero-copy">
          <p className="eyebrow">Protected moderation studio</p>
          <h1 className="display">
            Question moderation,
            <br />
            <span>with academic intent intact.</span>
          </h1>
          <p>
            Classify Bloom level, review a higher-order rewrite, summarize source material, or ask
            the local model for support.
          </p>
          <div className="teacher-hero-steps">
            <span>01 Classify</span>
            <span>02 Moderate</span>
            <span>03 Review</span>
          </div>
        </div>
        <video
          className="teacher-hero-video"
          src="/teacher-moderation-workflow.mp4"
          autoPlay
          loop
          muted
          playsInline
          controls
          preload="metadata"
        />
      </section>


      <section className="glass-card rounded-[26px] p-5 sm:p-7">
        <h2 className="section-title">Course material</h2>
        <p className="section-copy">Index a file or paste text into your authorized local corpus.</p>
        <div className="mt-6 grid gap-4">
          <label className="glass-pill block rounded-2xl border-dashed p-5 text-center">
            <span className="font-bold text-[#36577a] flex items-center justify-center gap-2">
              <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              {file ? file.name : "Choose a learning document"}
            </span>
            <input
              className="sr-only"
              type="file"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
          </label>
          <button className="btn-secondary" disabled={!!busy || !file} onClick={upload}>
            <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            Upload and index file
          </button>
          <textarea
            className="min-h-32"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste learning material to index..."
          />
          <button className="btn-primary" disabled={!!busy || !text.trim()} onClick={index}>
            Index pasted material
          </button>
        </div>
      </section>
      <section className="teacher-console glass-card rounded-[26px] p-5 sm:p-7">
        <div>
          <p className="eyebrow">Question workspace</p>
          <h2 className="section-title mt-3">Bring a question into focus.</h2>
        </div>
        <label className="teacher-question-label">
          Assessment question or academic text
          <textarea
            value={q}
            onChange={(e) => handleQuestionChange(e.target.value)}
            placeholder="e.g. Explain how formative assessment improves student learning."
          />
        </label>
        <div className="teacher-actions">
          <button className="btn-primary" disabled={!!busy || !q.trim()} onClick={() => run("classify")}>
            Classify level
          </button>
          <button
            className="btn-secondary"
            disabled={!!busy || !q.trim()}
            onClick={() => run("linguistic")}
          >
            Moderate question
          </button>
          <div className="teacher-rewrite-dropdown" ref={dropdownRef}>
            <button
              className="btn-secondary dropdown-trigger"
              ref={triggerRef}
              disabled={!!busy || !q.trim()}
              onClick={handleDropdownToggle}
            >
              <span>Target Level Rewrite</span>
              <svg 
                className={`dropdown-arrow ${dropdownOpen ? 'open' : ''}`} 
                viewBox="0 0 24 24" 
                fill="none" 
                stroke="currentColor" 
                strokeWidth="2" 
                strokeLinecap="round" 
                strokeLinejoin="round"
              >
                <polyline points="6 9 12 15 18 9" />
              </svg>
            </button>
            {dropdownOpen && (
              <div 
                className="dropdown-menu glass-card"
                style={{
                  position: 'fixed',
                  top: `${dropdownPosition.top}px`,
                  left: `${dropdownPosition.left}px`
                }}
              >
                <button 
                  className="dropdown-item"
                  onClick={() => handleTargetLevelSelect("Remember")}
                  disabled={!!busy}
                >
                  <span>Remember</span>
                  <span className="dropdown-item-sub">C1</span>
                </button>
                <button 
                  className="dropdown-item"
                  onClick={() => handleTargetLevelSelect("Understand")}
                  disabled={!!busy}
                >
                  <span>Understand</span>
                  <span className="dropdown-item-sub">C2</span>
                </button>
                <button 
                  className="dropdown-item"
                  onClick={() => handleTargetLevelSelect("Apply")}
                  disabled={!!busy}
                >
                  <span>Apply</span>
                  <span className="dropdown-item-sub">C3</span>
                </button>
                <button 
                  className="dropdown-item"
                  onClick={() => handleTargetLevelSelect("Analyze")}
                  disabled={!!busy}
                >
                  <span>Analyze</span>
                  <span className="dropdown-item-sub">C4</span>
                </button>
                <button 
                  className="dropdown-item"
                  onClick={() => handleTargetLevelSelect("Evaluate")}
                  disabled={!!busy}
                >
                  <span>Evaluate</span>
                  <span className="dropdown-item-sub">C5</span>
                </button>
                <button 
                  className="dropdown-item"
                  onClick={() => handleTargetLevelSelect("Create")}
                  disabled={!!busy}
                >
                  <span>Create</span>
                  <span className="dropdown-item-sub">C6</span>
                </button>
              </div>
            )}
          </div>
          <button className="btn-secondary" disabled={!!busy || !q.trim()} onClick={() => run("summary")}>
            Summarize
          </button>
          <button className="btn-quiet" disabled={!!busy || !q.trim()} onClick={() => run("qa")}>
            Ask model →
          </button>
        </div>
      </section>

      {(level || mod || linguisticMod || targetRewrite || answer || summary) && (
        <section className="teacher-results">
          {level && (
            <article className="glass-card teacher-result-card rounded-[22px] p-5">
              <p className="eyebrow">Bloom classification</p>
              <div className="teacher-level">
                <strong>{level}</strong>
                <span>
                  {bloom
                    ? `${Math.round(bloom.confidence * 100)}% confidence`
                    : "Classified locally"}
                </span>
              </div>
              <p className="section-copy">
                Use this level as the starting point for a proportionate assessment task.
              </p>
            </article>
          )}
          {mod && (
            <article className="glass-card teacher-result-card teacher-rewrite-card rounded-[22px] p-5">
              <p className="eyebrow">Higher level rewrite suggestion</p>
              <h2 className="section-title mt-2">Elevate to {mod.target_higher_level}</h2>
              <p className="teacher-reason">{mod.reason}</p>
              <div className="teacher-rewrite">
                <span>Suggested rewrite</span>
                <p>{mod.higher_level_rewrite}</p>
              </div>
            </article>
          )}
          {linguisticMod && (
            <article className="glass-card teacher-result-card rounded-[22px] p-5">
              <p className="eyebrow">Linguistic moderation</p>
              <div className="teacher-comparison">
                <div>
                  <span className="text-sm font-semibold text-[#66819e]">Original</span>
                  <p className="mt-2 text-sm leading-relaxed">{linguisticMod.original}</p>
                </div>
                <div className="text-center text-2xl text-[#416ce4]">→</div>
                <div>
                  <span className="text-sm font-semibold text-[#29795f]">Improved</span>
                  <p className="mt-2 text-sm leading-relaxed">{linguisticMod.improved}</p>
                </div>
              </div>
            </article>
          )}
          {targetRewrite && targetRewrite.rewritten && (
            <article className="glass-card teacher-result-card teacher-rewrite-card rounded-[22px] p-5">
              <p className="eyebrow">Target level rewrite</p>
              <div className="flex items-center gap-3 mt-2">
                <h2 className="section-title">Rewritten for {targetRewrite.target_level}</h2>
              </div>
              <div className="teacher-rewrite">
                <span>Original</span>
                <p className="text-sm text-gray-600">{targetRewrite.original}</p>
              </div>
              <div className="teacher-rewrite">
                <span>Rewritten</span>
                <p>{targetRewrite.rewritten}</p>
              </div>
            </article>
          )}
          {summary && (
            <article className="glass-card teacher-result-card rounded-[22px] p-5">
              <p className="eyebrow">Local summary</p>
              <p className="teacher-answer">{summary}</p>
            </article>
          )}
          {answer && (
            <article className="glass-card teacher-result-card rounded-[22px] p-5">
              <p className="eyebrow">Q&A with the model</p>
              <p className="teacher-answer">{answer}</p>
            </article>
          )}
        </section>
      )}

      {error && (
        <div className="mt-4 rounded-xl bg-red-50/70 p-3 text-sm text-[#a03d49]">
          <p>{error}</p>
          {!classifierLoaded && !busy && (
            <button type="button" className="btn-secondary mt-3" onClick={() => setClassifierLoadAttempt((attempt) => attempt + 1)}>
              Retry classifier loading
            </button>
          )}
        </div>
      )}

      {busy && <LoadingComponent label={busy} hint={loadingHint} showGame={true} onCancel={cancelOperation} />}
      {showSuccess && <LoadingComponent label="" hint="Operation completed successfully" success={true} successMessage={success.message} finalElapsed={success.elapsed} />}
    </main>
  );
}

export default TeacherStudio;
