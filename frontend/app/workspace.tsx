"use client";

import dynamic from "next/dynamic";

import { FormEvent, useCallback, useEffect, useRef, useState } from "react";

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

const clips = [
  ["Initial_Scene_1.mp4", "Settle in", "A calm place to begin."],
  ["Initial_Scene_2.mp4", "Find the thread", "Make ideas connect."],
  ["Initial_scene_4.mp4", "Learn actively", "Bring your notes and curiosity."],
  ["Initial_scene_5.mp4", "Build confidence", "Turn study into progress."],
  ["Initial_scene_6.mp4", "Keep going", "Move at your own pace."],
  ["Initial_scene_final.mp4", "Make it yours", "Private and on-device."],
] as const;

type Role = "student" | "teacher";
type Msg = { role: "user" | "assistant"; content: string };
type Bloom = { level: string; confidence: number };
type Moderation = {
  bloom_level: string;
  target_higher_level: string;
  reason: string;
  higher_level_rewrite: string;
};


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

function formatElapsed(seconds: number) {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return mins > 0 ? `${mins}:${secs.toString().padStart(2, "0")}` : `${secs}s`;
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

function WaitingGame({ active }: { active: boolean }) {
  const [score, setScore] = useState(0);
  const [streak, setStreak] = useState(0);
  const [bestScore, setBestScore] = useState(0);
  const [objects, setObjects] = useState<Array<{ id: number; x: number; type: string; duration: number }>>([]);
  const objectIdCounter = useRef(0);
  const spawnIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const prefersReducedMotion = useRef(false);

  // Check for reduced motion preference
  useEffect(() => {
    prefersReducedMotion.current = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }, []);

  // Game difficulty based on score
  const getSpawnRate = () => {
    if (score < 15) return 1800;
    if (score < 35) return 1400;
    return 1000;
  };

  const getFallDuration = () => {
    if (score < 15) return 4 + Math.random() * 2;
    if (score < 35) return 3 + Math.random() * 1.5;
    return 2 + Math.random() * 1;
  };

  // Spawn new objects
  const spawnObject = useCallback(() => {
    if (!active || prefersReducedMotion.current) return;

    const types = ['star', 'dot', 'dot', 'book'];
    const type = types[Math.floor(Math.random() * types.length)];
    const isFocusToken = Math.random() < 0.12; // 12% chance for focus token

    const newObject = {
      id: objectIdCounter.current++,
      x: Math.random() * 80 + 10, // 10-90% width
      type: isFocusToken ? 'focus' : type,
      duration: getFallDuration()
    };

    setObjects(prev => [...prev.slice(-5), newObject]); // Keep max 6 objects on screen
  }, [active, score]);

  // Spawn loop
  useEffect(() => {
    if (!active || prefersReducedMotion.current) {
      if (spawnIntervalRef.current) {
        clearInterval(spawnIntervalRef.current);
        spawnIntervalRef.current = null;
      }
      return;
    }

    const spawnRate = getSpawnRate();
    spawnIntervalRef.current = setInterval(spawnObject, spawnRate);

    return () => {
      if (spawnIntervalRef.current) {
        clearInterval(spawnIntervalRef.current);
        spawnIntervalRef.current = null;
      }
    };
  }, [active, score, spawnObject]);

  // Handle object catch
  const handleCatch = useCallback((objectId: number, type: string) => {
    if (!active) return; // Prevent catching after game stops
    
    setObjects(prev => prev.filter(obj => obj.id !== objectId));
    
    const points = type === 'focus' ? 3 : 1;
    setScore(prev => prev + points);
    setStreak(prev => prev + 1);
  }, [active]);

  // Update best score when score changes
  useEffect(() => {
    if (score > bestScore) {
      setBestScore(score);
    }
  }, [score, bestScore]);

  // Handle object miss (animation end)
  const handleMiss = () => {
    setStreak(0);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (spawnIntervalRef.current) {
        clearInterval(spawnIntervalRef.current);
      }
    };
  }, []);

  // Reset game when not active
  useEffect(() => {
    if (!active) {
      setScore(0);
      setStreak(0);
      setObjects([]);
      if (spawnIntervalRef.current) {
        clearInterval(spawnIntervalRef.current);
        spawnIntervalRef.current = null;
      }
    }
  }, [active]);

  if (!active) return null;

  if (prefersReducedMotion.current) {
    return (
      <div className="waiting-game-container">
        <p className="waiting-game-message">Processing your request...</p>
      </div>
    );
  }

  const icons = {
   star: '✦',
    dot: '•',
    book: '📖',
    focus: '⚡'
  };

  return (
    <div className="waiting-game-container">
      <div className="waiting-game-header">
        <span className="waiting-game-title">Focus Catch</span>
        <div className="waiting-game-stats">
          <span>Score {score}</span>
          {streak > 1 && <span>• Streak {streak}</span>}
          {bestScore > 0 && <span>• Best {bestScore}</span>}
        </div>
      </div>
      <div className="waiting-game-area">
        {objects.map(obj => (
          <button
            key={obj.id}
            className="waiting-game-object"
            style={{
              left: `${obj.x}%`,
              animationDuration: `${obj.duration}s`,
            }}
            onClick={() => handleCatch(obj.id, obj.type)}
            onAnimationEnd={handleMiss}
            aria-label={`Catch ${obj.type}`}
          >
            {icons[obj.type as keyof typeof icons]}
          </button>
        ))}
        <div className="waiting-game-catcher">
          <span>CATCH</span>
        </div>
      </div>
      <p className="waiting-game-instruction">Catch the falling ideas</p>
    </div>
  );
}

function Loading({
  label,
  hint = "Your local model is preparing a grounded response.",
  success = false,
  successMessage = "",
  finalElapsed = 0,
  showGame = false,
  onCancel,
}: {
  label: string;
  hint?: string;
  success?: boolean;
  successMessage?: string;
  finalElapsed?: number;
  showGame?: boolean;
  onCancel?: () => void;
}) {
  const elapsed = useElapsed(!success);

  return (
    <div className="loading-dialog-backdrop" role="dialog" aria-modal="true" aria-label={success ? successMessage : label}>
      <section className={`loading-dialog ${showGame && !success ? 'loading-dialog-with-game' : ''}`}>
        {!success && (
          <button
            type="button"
            className="loading-dialog-close"
            onClick={() => onCancel?.() }
            aria-label="Close loading dialog"
            title="Close"
          >
            <svg aria-hidden="true" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M6 6 18 18M18 6 6 18" /></svg>
          </button>
        )}
        {success ? (
          <div className="loading-success">
            <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="20 6 9 17 4 12" />
            </svg>
          </div>
        ) : (
          <span className="loading-orbit">
            <i />
          </span>
        )}
        <p>{success ? successMessage : label}</p>
        {hint && <small>{hint}</small>}
        <div className="loading-elapsed">
          Elapsed <b>{formatElapsed(success ? finalElapsed : elapsed)}</b>
        </div>
        {showGame && !success && <WaitingGame active={!success} />}
      </section>
    </div>
  );
}

function ChatLoading({ label }: { label: string }) {
  const elapsed = useElapsed(true);

  return (
    <article className="student-message assistant student-message-loading">
      <p className="student-message-label">Assistant</p>
      <span className="typing-dots" aria-label="Loading">
        <i />
        <i />
        <i />
      </span>
      <p className="student-message-sources">
        {label} · <b>{formatElapsed(elapsed)}</b>
      </p>
    </article>
  );
}

function isIndexingBusy(busy: string) {
  return busy.includes("Index") || busy.includes("Upload");
}

function isModelBusy(busy: string) {
  return (
    busy.includes("model") ||
    busy.includes("summary") ||
    busy.includes("reply") ||
ess.elapsed} />}
    </main>
  );
}

const TeacherStudio = dynamic(() => import("./teacher-studio"), {
  loading: () => (
    <main className="page-wrap fade-in">
      <section className="glass-card rounded-[26px] p-6" aria-live="polite">
        <p className="eyebrow">Teacher workspace</p>
        <p className="section-copy mt-3">Loading teacher tools…</p>
      </section>
    </main>
  ),
});

export default function Workspace({ role }: { role: Role }) {
  const [token, setToken] = useState("");
  const [code, setCode] = useState("");
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [q, setQ] = useState("");
  const [history, setHistory] = useState<Msg[]>([]);
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
  const [clip, setClip] = useState<(typeof clips)[number] | null>(null);
  const [success, setSuccess] = useState<{ show: boolean; message: string; elapsed: number }>({ show: false, message: "", elapsed: 0 });
  const operationElapsed = useElapsed(!!busy);

  async function call(path: string, body: object) {
    const active = operationController.current;
    const r = await apiFetch(`${API}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(active ? { "X-Operation-ID": active.id } : {}),
      },
      body: JSON.stringify(body),
      signal: active?.controller.signal,
    });
    const d = await r.json();
    if (!r.ok) throw Error(d.detail || "The local service could not complete this request.");
    return d;
  }

  async function login(e: FormEvent) {
    e.preventDefault();
    setBusy("Opening workspace...");
    setError("");
    try {
      const session = await call("/auth/session", { role, access_code: code });
      setToken(session.access_token);
    } catch (e) {
      setError(String(e).replace("Error: ", ""));
    } finally {
      setBusy("");
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
      await call("/documents/index", {
        text,
        name: "study-notes",
        scope: "public",
        content_type: "study_material",
      });
      setSuccess({ show: true, message: "Indexing complete", elapsed: operationElapsed });
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) setError(String(e).replace("Error: ", ""));
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

  async function ask(summary = false) {
    if (!q.trim()) return;
    const operation = createOperation();
    operationController.current = operation;
    setBusy(summary ? "Preparing a local summary" : "Loading the local model and writing a reply");
    setError("");
    try {
      const d = await call(summary ? "/summarize" : "/chat", {
        question: q,
        scope: "public",
        top_k: 4,
        summary,
        history,
      });
      setHistory((h) => [
        ...h,
        { role: "user", content: q },
        { role: "assistant", content: d.answer },
      ]);
      setQ("");
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) setError(String(e).replace("Error: ", ""));
    } finally {
      const stillCurrent = operationController.current === operation;
      if (stillCurrent) operationController.current = null;
      if (stillCurrent) setBusy("");
    }
  }

  const form = (
    <form className="mt-7 grid gap-4" onSubmit={login}>
      <label>
        Local access code
        <input
          value={code}
          onChange={(e) => setCode(e.target.value)}
          type="password"
          placeholder="Enter access code"
          autoFocus
        />
      </label>
      <button className="btn-primary" disabled={!!busy}>
        {busy ? "Opening workspace..." : "Continue to workspace →"}
      </button>
    </form>
  );

  if (!token && role === "student") {
    return (
      <main className="page-wrap fade-in student-access-page">
        <section className="student-access-layout">
          <div className="student-access-intro">
            <p className="eyebrow">Student learning studio</p>
            <h1 className="display">
              A quieter space
              <br />
              to make progress.
            </h1>
            <p>
              Ask questions, bring your own notes, and turn focused study into understanding—privately,
              on your computer.
            </p>
            <div className="student-access-stills">
              <img src="/student-videos/Initial scene_3.jpeg" alt="A reflective study moment" />
              <img src="/student-videos/Initial_scene_7.jpeg" alt="A calm learning environment" />
            </div>
          </div>
          <section className="glass-card student-access-card">
            <span className="status">Private / offline</span>
            <h2 className="display">Enter your learning space</h2>
            <p className="section-copy">
              Use your locally configured access code. Nothing leaves this device.
            </p>
            {form}
            {error && (
              <p className="mt-4 rounded-xl bg-red-50/70 p-3 text-sm text-[#a03d49]">{error}</p>
            )}
          </section>
        </section>
      </main>
    );
  }

  if (!token && role === "teacher") {
    return (
      <main className="page-wrap fade-in student-access-page">
        <section className="student-access-layout teacher-access-layout">
          <div className="student-access-intro">
            <p className="eyebrow">Protected moderation studio</p>
            <h1 className="display">
              Assess with
              <br />
              clearer intent.
            </h1>
            <p>
              Classify question level, review a higher-order rewrite, and maintain sound assessment
              design—on your local system.
            </p>
            <div className="teacher-access-preview">
              <video
                src="/teacher-moderation-workflow.mp4"
                autoPlay
                loop
                muted
                playsInline
                preload="metadata"
              />
            </div>
          </div>
          <section className="glass-card student-access-card">
            <span className="status">Teacher / local</span>
            <h2 className="display">Enter teacher workspace</h2>
            <p className="section-copy">
              Use your teacher access code to open the protected moderation tools.
            </p>
            {form}
            {error && (
              <p className="mt-4 rounded-xl bg-red-50/70 p-3 text-sm text-[#a03d49]">{error}</p>
            )}
          </section>
        </section>
      </main>
    );
  }

  if (role === "teacher") return <TeacherStudio token={token} LoadingComponent={Loading} />;

  const showIndexingOverlay = !!busy && isIndexingBusy(busy);
  const showModelLoading = !!busy && isModelBusy(busy);
  const showSuccess = success.show && !busy;

  return (
    <main className="page-wrap fade-in">
      <header className="mb-8">
        <p className="eyebrow">Student learning studio</p>
        <h1 className="display mt-3 text-4xl font-extrabold">Student workspace</h1>
        <p className="section-copy">
          Build understanding privately with a local model or your own material.
        </p>
      </header>

      <section className="learning-reel">
        <div className="learning-reel-heading">
          <div>
            <p className="eyebrow">Your learning rhythm</p>
            <h2 className="section-title">Small moments, real momentum.</h2>
          </div>
          <p>Select a moment to watch it in full.</p>
        </div>
        <div className="learning-reel-track">
          {clips.map((item, i) => (
            <button
              type="button"
              className="learning-reel-card learning-reel-button"
              key={item[0]}
              onClick={() => setClip(item)}
              aria-label={`Watch ${item[1]}`}
            >
              <video
                src={`/student-videos/${item[0]}`}
                autoPlay
                loop
                muted
                playsInline
                preload={i < 2 ? "auto" : "metadata"}
              />
              <div className="learning-reel-overlay">
                <span>0{i + 1}</span>
                <h3>{item[1]}</h3>
                <p>{item[2]}</p>
                <b>Watch ➡</b>
              </div>
            </button>
          ))}
        </div>
      </section>

      <div className="grid gap-5 lg:grid-cols-[.92fr_1.08fr]">
        <section className="glass-card lift rounded-[26px] p-5 sm:p-7">
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

        <section className="glass-card lift student-chat-card rounded-[26px] p-5 sm:p-7">
          <h2 className="section-title">Ask anything</h2>
          <div className="student-chat-thread" aria-live="polite">
            {history.length === 0 ? (
              <p className="student-chat-empty">Your local conversations stay in this session.</p>
            ) : (
              history.map((m, i) => (
                <article key={i} className={`student-message ${m.role}`}>
                  <p className="student-message-label">{m.role === "user" ? "You" : "Assistant"}</p>
                  <p className="student-message-content">{m.content}</p>
                </article>
              ))
            )}
            {showModelLoading && <ChatLoading label={busy} />}
          </div>
          <div className="mt-6 grid gap-4">
            <label>
              Question or text to summarize
              <textarea
                className="mt-1 min-h-32"
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="Ask a question or paste text to summarize..."
              />
            </label>
            <div className="flex flex-wrap gap-2">
              <button className="btn-primary" disabled={!!busy || !q.trim()} onClick={() => ask()}>
                Send
              </button>
              <button className="btn-secondary" disabled={!!busy || !q.trim()} onClick={() => ask(true)}>
                Summarize
              </button>
            </div>
          </div>
        </section>
      </div>

      {clip && (
        <div
          className="reel-dialog-backdrop"
          role="dialog"
          aria-modal="true"
          aria-label={clip[1]}
          onClick={() => setClip(null)}
        >
          <section className="reel-dialog" onClick={(e) => e.stopPropagation()}>
            <button className="reel-close" onClick={() => setClip(null)} aria-label="Close video">
              ×
            </button>
            <video src={`/student-videos/${clip[0]}`} controls autoPlay playsInline />
            <div>
              <p className="eyebrow">Learning moment</p>
              <h2 className="section-title">{clip[1]}</h2>
              <p className="section-copy">{clip[2]}</p>
            </div>
          </section>
        </div>
      )}

      {error && (
        <p className="mt-4 rounded-xl bg-red-50/70 p-3 text-sm text-[#a03d49]">{error}</p>
      )}

      {showIndexingOverlay && (
        <Loading
          label={busy}
          hint="Preparing your material for source-grounded answers."
          showGame={true}
          onCancel={cancelOperation}
        />
      )}
      {showSuccess && (
        <Loading
          label=""
          hint="Operation completed successfully"
          success={true}
          successMessage={success.message}
          finalElapsed={success.elapsed}
        />
      )}
    </main>
  );
}

