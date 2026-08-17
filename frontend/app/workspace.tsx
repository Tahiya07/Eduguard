"use client";

import { FormEvent, useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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

function Loading({
  label,
  hint = "Your local model is preparing a grounded response.",
}: {
  label: string;
  hint?: string;
}) {
  const elapsed = useElapsed(true);

  return (
    <div className="loading-dialog-backdrop" role="status" aria-live="polite">
      <section className="loading-dialog">
        <span className="loading-orbit">
          <i />
        </span>
        <p>{label}</p>
        <small>{hint}</small>
        <div className="loading-elapsed">
          Elapsed <b>{formatElapsed(elapsed)}</b>
        </div>
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
    busy.includes("Working")
  );
}

function TeacherStudio({ token }: { token: string }) {
  const [q, setQ] = useState("");
  const [bloom, setBloom] = useState<Bloom | null>(null);
  const [mod, setMod] = useState<Moderation | null>(null);
  const [answer, setAnswer] = useState("");
  const [summary, setSummary] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  async function req(path: string, body: object) {
    const r = await fetch(`${API}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(body),
    });
    const d = await r.json();
    if (!r.ok) throw Error(d.detail || "The local service could not complete this request.");
    return d;
  }

  async function run(action: "classify" | "moderate" | "qa" | "summary") {
    if (!q.trim()) return;

    const labels = {
      classify: "Classifying the question",
      moderate: "Preparing moderation guidance",
      qa: "Loading the local model and writing a reply",
      summary: "Preparing a local summary",
    } as const;

    setBusy(labels[action]);
    setError("");

    try {
      if (action === "classify") {
        const d = await req("/teacher/exam/classify", { question: q });
        setBloom({ level: d.level, confidence: d.confidence });
      }
      if (action === "moderate") {
        const d = await req("/teacher/exam/moderate", { question: q });
        setBloom({ level: d.bloom.level, confidence: d.bloom.confidence });
        setMod(d.moderation);
      }
      if (action === "qa") {
        setAnswer((await req("/qa", { question: q, scope: "public", top_k: 4 })).answer);
      }
      if (action === "summary") {
        setSummary((await req("/summarize", { question: q, scope: "public", top_k: 4 })).answer);
      }
    } catch (e) {
      setError(String(e).replace("Error: ", ""));
    } finally {
      setBusy("");
    }
  }

  const level = mod?.bloom_level || bloom?.level;
  const loadingHint =
    busy === "Classifying the question" || busy === "Preparing moderation guidance"
      ? "Reviewing the question against local Bloom guidance."
      : "Your local model is preparing a grounded response.";

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

      <section className="teacher-console glass-card rounded-[26px] p-5 sm:p-7">
        <div>
          <p className="eyebrow">Question workspace</p>
          <h2 className="section-title mt-3">Bring a question into focus.</h2>
        </div>
        <label className="teacher-question-label">
          Assessment question or academic text
          <textarea
            value={q}
            onChange={(e) => setQ(e.target.value)}
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
            onClick={() => run("moderate")}
          >
            Moderate & rewrite
          </button>
          <button className="btn-secondary" disabled={!!busy || !q.trim()} onClick={() => run("summary")}>
            Summarize
          </button>
          <button className="btn-quiet" disabled={!!busy || !q.trim()} onClick={() => run("qa")}>
            Ask model →
          </button>
        </div>
      </section>

      {(level || mod || answer || summary) && (
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
              <p className="eyebrow">Moderation recommendation</p>
              <h2 className="section-title mt-2">Elevate to {mod.target_higher_level}</h2>
              <p className="teacher-reason">{mod.reason}</p>
              <div className="teacher-rewrite">
                <span>Suggested rewrite</span>
                <p>{mod.higher_level_rewrite}</p>
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
        <p className="mt-4 rounded-xl bg-red-50/70 p-3 text-sm text-[#a03d49]">{error}</p>
      )}

      {busy && <Loading label={busy} hint={loadingHint} />}
    </main>
  );
}

export default function Workspace({ role }: { role: Role }) {
  const [token, setToken] = useState("");
  const [code, setCode] = useState("");
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [q, setQ] = useState("");
  const [history, setHistory] = useState<Msg[]>([]);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [clip, setClip] = useState<(typeof clips)[number] | null>(null);

  async function call(path: string, body: object) {
    const r = await fetch(`${API}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(body),
    });
    const d = await r.json();
    if (!r.ok) throw Error(d.detail || "The local service could not complete this request.");
    return d;
  }

  async function login(e: FormEvent) {
    e.preventDefault();
    setBusy("login");
    setError("");
    try {
      setToken((await call("/auth/session", { role, access_code: code })).access_token);
    } catch (e) {
      setError(String(e).replace("Error: ", ""));
    } finally {
      setBusy("");
    }
  }

  async function index() {
    if (!text.trim()) return;
    setBusy("Indexing your material");
    setError("");
    try {
      await call("/documents/index", {
        text,
        name: "study-notes",
        scope: "public",
        content_type: "study_material",
      });
    } catch (e) {
      setError(String(e).replace("Error: ", ""));
    } finally {
      setBusy("");
    }
  }

  async function upload() {
    if (!file) return;
    setBusy("Uploading and indexing your file");
    setError("");
    try {
      const f = new FormData();
      f.set("file", file);
      f.set("scope", "public");
      f.set("content_type", "study_material");
      const r = await fetch(`${API}/documents/upload`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: f,
      });
      if (!r.ok) throw Error("Upload failed");
    } catch (e) {
      setError(String(e).replace("Error: ", ""));
    } finally {
      setBusy("");
    }
  }

  async function ask(summary = false) {
    if (!q.trim()) return;
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
      setError(String(e).replace("Error: ", ""));
    } finally {
      setBusy("");
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

  if (!token) {
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

  if (role === "teacher") return <TeacherStudio token={token} />;

  const showIndexingOverlay = !!busy && isIndexingBusy(busy);
  const showModelLoading = !!busy && isModelBusy(busy);

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
                <b>Watch →</b>
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
              <span className="font-bold text-[#36577a]">
                {file ? file.name : "Choose a learning document"}
              </span>
              <input
                className="sr-only"
                type="file"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
            </label>
            <button className="btn-secondary" disabled={!!busy || !file} onClick={upload}>
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
        />
      )}
    </main>
  );
}
