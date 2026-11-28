"use client";

import dynamic from "next/dynamic";
import Image from "next/image";
import {
	type FormEvent,
	type KeyboardEvent, 
	useCallback,
	useEffect,
	useRef,
	useState,
} from "react";
import ErrorModal, { errorTitle } from "./error-modal";
import { clearSession, saveSession, useSession } from "./session";
import type { HeroClip } from "./video-hero";
import VideoHero from "./video-hero";
import WaitingGame from "./waiting-game";
import { type OperationProgress, useOperationProgress } from "./operation-progress";
import PasswordInput from "./password-input";
import FilePicker from "./file-picker";

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
			if (
				init?.signal?.aborted ||
				(error instanceof DOMException && error.name === "AbortError")
			)
				throw error;

			if (attempt === retries) {
				throw lastError;
			}

			// Backend may still be initializing. Wait progressively,
			// but never block the UI itself.
			await new Promise<void>((resolve, reject) => {
				const signal = init?.signal;
				const timeout = window.setTimeout(
					() => {
						signal?.removeEventListener("abort", abort);
						resolve();
					},
					Math.min(1000 + attempt * 500, 3000),
				);
				const abort = () => {
					window.clearTimeout(timeout);
					reject(
						new DOMException("The operation was cancelled.", "AbortError"),
					);
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

const STUDENT_CLIPS: HeroClip[] = clips.map(([file, title, caption]) => ({
	src: `/student-videos/${file}`,
	title,
	caption,
}));

function scrollToId(id: string, focusId?: string) {
	document
		.getElementById(id)
		?.scrollIntoView({ behavior: "smooth", block: "start" });
	if (focusId)
		window.setTimeout(
			() => document.getElementById(focusId)?.focus({ preventScroll: true }),
			350,
		);
}

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

function cancelBackendOperation(
	operation: { id: string; controller: AbortController } | null,
	token: string,
) {
	if (!operation) return;
	operation.controller.abort();
	void fetch(
		`${API}/operations/cancel?operation_id=${encodeURIComponent(operation.id)}`,
		{
			method: "POST",
			headers: token ? { Authorization: `Bearer ${token}` } : {},
		},
	).catch(() => {
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

function Loading({
	label,
	hint = "Your local model is preparing a grounded response.",
	success = false,
	successMessage = "",
	finalElapsed = 0,
	showGame = false,
	onCancel,
	progress = null,
}: {
	label: string;
	hint?: string;
	success?: boolean;
	successMessage?: string;
	finalElapsed?: number;
	showGame?: boolean;
	onCancel?: () => void;
	progress?: OperationProgress | null;
}) {
	const elapsed = useElapsed(!success);

	return (
		<div
			className="loading-dialog-backdrop"
			role="dialog"
			aria-modal="true"
			aria-label={success ? successMessage : label}
		>
			<section
				className={`loading-dialog ${showGame && !success ? "loading-dialog-with-game" : ""}`}
			>
				{!success && (
					<button
						type="button"
						className="loading-dialog-close"
						onClick={() => onCancel?.()}
						aria-label="Close loading dialog"
						title="Close"
					>
						<svg
							aria-hidden="true"
							viewBox="0 0 24 24"
							width="16"
							height="16"
							fill="none"
							stroke="currentColor"
							strokeWidth="2"
							strokeLinecap="round"
						>
							<path d="M6 6 18 18M18 6 6 18" />
						</svg>
					</button>
				)}
				{success ? (
					<div className="loading-success">
						<svg
							aria-hidden="true"
							viewBox="0 0 24 24"
							fill="none"
							stroke="currentColor"
							strokeWidth="3"
							strokeLinecap="round"
							strokeLinejoin="round"
						>
							<polyline points="20 6 9 17 4 12" />
						</svg>
					</div>
				) : (
					<span className="loading-orbit">
						<i />
					</span>
				)}
				<p>{success ? successMessage : label}</p>
				{!success && progress?.message ? (
					<ProgressLine progress={progress} />
				) : (
					hint && <small>{hint}</small>
				)}
				<div className="loading-elapsed">
					Elapsed <b>{formatElapsed(success ? finalElapsed : elapsed)}</b>
				</div>
				{showGame && !success && <WaitingGame active={!success} />}
			</section>
		</div>
	);
}

/** The step a long operation is on, with a bar when the step count is known. */
function ProgressLine({ progress }: { progress: OperationProgress }) {
	const pct =
		progress.total && progress.step
			? Math.min(100, Math.round((100 * progress.step) / progress.total))
			: null;
	return (
		<div className="loading-progress" aria-live="polite">
			<small>{progress.message}</small>
			{pct !== null && (
				<span className="loading-progress-bar" aria-hidden="true">
					<i style={{ width: `${pct}%` }} />
				</span>
			)}
		</div>
	);
}

function ChatLoading({
	label,
	progress = null,
}: {
	label: string;
	progress?: OperationProgress | null;
}) {
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
				{progress?.message || label} · <b>{formatElapsed(elapsed)}</b>
			</p>
		</article>
	);
}

function isIndexingBusy(busy: string) {
	return busy.includes("Index") || busy.includes("Upload");
}

function isModelBusy(busy: string) {
	return (
		busy.includes("model") || busy.includes("summary") || busy.includes("reply")
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
	// Restore this tab's session so navigating Home <-> workspace keeps the same server workspace.
	const session = useSession();
	useEffect(() => {
		setToken(session && session.role === role ? session.token : "");
	}, [session, role]);
	const [code, setCode] = useState("");
	const [text, setText] = useState("");
	const [file, setFile] = useState<File | null>(null);
	const [q, setQ] = useState("");
	const [history, setHistory] = useState<Msg[]>([]);
	const [mode, setMode] = useState<"ask" | "summary">("ask");
	const [materials, setMaterials] = useState<{ id: string; name: string }[]>(
		[],
	);
	const threadRef = useRef<HTMLDivElement>(null);
	const [busy, setBusy] = useState("");
	const operationController = useRef<ReturnType<typeof createOperation> | null>(
		null,
	);
	const currentOperationId = useCallback(() => operationController.current?.id, []);
	const progress = useOperationProgress(API, token, currentOperationId, !!busy);
	const cancelOperation = useCallback(() => {
		const active = operationController.current;
		operationController.current = null;
		cancelBackendOperation(active, token);
		setBusy("");
		setSuccess({ show: false, message: "", elapsed: 0 });
	}, [token]);
	const [error, setError] = useState("");
	const [success, setSuccess] = useState<{
		show: boolean;
		message: string;
		elapsed: number;
	}>({ show: false, message: "", elapsed: 0 });
	const operationElapsed = useElapsed(!!busy);

	// Keep the newest message in view (declared before any early return to keep hook order stable).
	useEffect(() => {
		const el = threadRef.current;
		if (el) el.scrollTop = el.scrollHeight;
	}, [history.length, busy]);

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
		if (r.status === 401 && token) {
			clearSession();
			throw Error("Your session expired. Please enter your access code again.");
		}
		if (!r.ok)
			throw Error(
				d.detail || "The local service could not complete this request.",
			);
		return d;
	}

	async function login(e: FormEvent) {
		e.preventDefault();
		setBusy("Opening workspace...");
		setError("");
		try {
			const started = await call("/auth/session", { role, access_code: code });
			setToken(started.access_token);
			saveSession({ token: started.access_token, role });
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
			setMaterials((m) => [
				...m,
				{
					id: crypto.randomUUID(),
					name: `Pasted notes (${text.trim().split(/\s+/).length} words)`,
				},
			]);
			setText("");
			setSuccess({
				show: true,
				message: "Indexing complete",
				elapsed: operationElapsed,
			});
		} catch (e) {
			if (!(e instanceof DOMException && e.name === "AbortError"))
				setError(String(e).replace("Error: ", ""));
		} finally {
			const stillCurrent = operationController.current === operation;
			if (stillCurrent) operationController.current = null;
			const finalElapsed = operationElapsed;
			if (stillCurrent) {
				setBusy("");
				if (!operation.controller.signal.aborted)
					setSuccess((prev) => ({ ...prev, elapsed: finalElapsed }));
				setTimeout(
					() => setSuccess({ show: false, message: "", elapsed: 0 }),
					2000,
				);
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
				headers: {
					Authorization: `Bearer ${token}`,
					"X-Operation-ID": operation.id,
				},
				body: f,
				signal: operation.controller.signal,
			});
			if (r.status === 401) {
				clearSession();
				throw Error(
					"Your session expired. Please enter your access code again.",
				);
			}
			if (!r.ok) {
				// Show the service's own reason (e.g. OCR missing) instead of a generic message.
				const detail = await r
					.json()
					.then((d) => d?.detail)
					.catch(() => null);
				throw Error(
					detail ||
						"Upload failed. Supported files: PDF, TXT, MD, CSV, JPG and PNG.",
				);
			}
			setMaterials((m) => [...m, { id: crypto.randomUUID(), name: file.name }]);
			setFile(null);
			setSuccess({
				show: true,
				message: "Indexing complete",
				elapsed: operationElapsed,
			});
		} catch (e) {
			if (
				operationController.current === operation &&
				!(e instanceof DOMException && e.name === "AbortError")
			)
				setError(String(e).replace("Error: ", ""));
		} finally {
			const stillCurrent = operationController.current === operation;
			if (stillCurrent) operationController.current = null;
			const finalElapsed = operationElapsed;
			if (stillCurrent) {
				setBusy("");
				if (!operation.controller.signal.aborted)
					setSuccess((prev) => ({ ...prev, elapsed: finalElapsed }));
				setTimeout(
					() => setSuccess({ show: false, message: "", elapsed: 0 }),
					2000,
				);
			}
		}
	}

	async function clearNotes() {
		setError("");
		try {
			const r = await apiFetch(`${API}/documents/clear?scope=public`, {
				method: "POST",
				headers: { Authorization: `Bearer ${token}` },
			});
			if (r.status === 401) {
				clearSession();
				throw Error(
					"Your session expired. Please enter your access code again.",
				);
			}
			if (!r.ok)
				throw Error("Could not remove the material. Please try again.");
			setMaterials([]);
			// Earlier answers came from the removed notes; keeping them as context made the model repeat them.
			setHistory([]);
			setSuccess({
				show: true,
				message: "Notes removed - new conversation started",
				elapsed: 0,
			});
			setTimeout(
				() => setSuccess({ show: false, message: "", elapsed: 0 }),
				1500,
			);
		} catch (e) {
			setError(e instanceof Error ? e.message : String(e));
		}
	}

	async function ask(summary = false) {
		if (!q.trim()) return;
		const operation = createOperation();
		operationController.current = operation;
		setBusy(
			summary
				? "Preparing a local summary"
				: "Loading the local model and writing a reply",
		);
		setError("");
		try {
			const d = await call(summary ? "/summarize" : "/chat", {
				question: q,
				scope: "public",
				top_k: 4,
				summary,
				// The API accepts at most 8 prior turns (the model uses the last 6), each up to 4,000 chars.
				// Sending the whole conversation made every message after the 4th exchange fail with HTTP 422.
				history: history
					.slice(-6)
					.map((m) => ({ role: m.role, content: m.content.slice(0, 4000) })),
			});
			setHistory((h) => [
				...h,
				{ role: "user", content: q },
				{ role: "assistant", content: d.answer },
			]);
			setQ("");
		} catch (e) {
			if (!(e instanceof DOMException && e.name === "AbortError"))
				setError(String(e).replace("Error: ", ""));
		} finally {
			const stillCurrent = operationController.current === operation;
			if (stillCurrent) operationController.current = null;
			if (stillCurrent) setBusy("");
		}
	}

	const form = (
		<form className="mt-7 grid gap-4" onSubmit={login}>
			<label htmlFor="access-code">
				Local access code
				<PasswordInput id="access-code" value={code} onChange={setCode} autoFocus />
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
							Ask questions, bring your own notes, and turn focused study into
							understanding—privately, on your computer.
						</p>
						<div className="student-access-stills">
							<Image
								src="/student-videos/Initial scene_3.jpeg"
								alt="A reflective study moment"
							/>
							<Image
								src="/student-videos/Initial_scene_7.jpeg"
								alt="A calm learning environment"
							/>
						</div>
					</div>
					<section className="glass-card student-access-card">
						<span className="status">Private / offline</span>
						<h2 className="display">Enter your learning space</h2>
						<p className="section-copy">
							Use your locally configured access code. Nothing leaves this
							device.
						</p>
						{form}
						{error && (
							<ErrorModal
								title={errorTitle(error)}
								message={error}
								onClose={() => setError("")}
							/>
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
							Classify question level, review a higher-order rewrite, and
							maintain sound assessment design—on your local system.
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
							Use your teacher access code to open the protected moderation
							tools.
						</p>
						{form}
						{error && (
							<ErrorModal
								title={errorTitle(error)}
								message={error}
								onClose={() => setError("")}
							/>
						)}
					</section>
				</section>
			</main>
		);
	}

	if (role === "teacher")
		return <TeacherStudio token={token} LoadingComponent={Loading} />;

	const showIndexingOverlay = !!busy && isIndexingBusy(busy);
	const showModelLoading = !!busy && isModelBusy(busy);
	const showSuccess = success.show && !busy;
	const isSummary = mode === "summary";
	const examples = [
		"What is the difference between TCP and UDP?",
		"Explain how a hash table handles collisions.",
		"Explain opportunity cost with an example.",
	];

	function send() {
		if (!busy && q.trim()) void ask(isSummary);
	}

	// Ask: Enter sends, Shift+Enter adds a line. Summarize: long pasted text, so Ctrl+Enter sends.
	function onComposerKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
		if (e.key !== "Enter") return;
		if (isSummary ? e.ctrlKey || e.metaKey : !e.shiftKey) {
			e.preventDefault();
			send();
		}
	}

	return (
		<main className="page-wrap fade-in">
			<VideoHero
				eyebrow="Student learning studio"
				title={
					<>
						A quieter space
						<br />
						<span>to make progress.</span>
					</>
				}
				description="Ask study questions, summarize your reading and bring your own notes. The assistant runs privately on this computer, so nothing you type leaves it."
				actions={
					<>
						<button
							type="button"
							className="btn-primary"
							onClick={() => scrollToId("study-assistant", "student-input")}
						>
							Start asking →
						</button>
						<button
							type="button"
							className="btn-secondary"
							onClick={() => scrollToId("your-notes")}
						>
							Add your notes
						</button>
					</>
				}
				trust={["Private", "Offline", "On this device"]}
				clips={STUDENT_CLIPS}
			/>

			<ol className="guide-steps" aria-label="How to use the workspace">
				<li>
					<b>1</b>
					<div>
						<strong>Ask or summarize</strong>
						<span>Type a question, or switch to Summarize and paste text.</span>
					</div>
				</li>
				<li>
					<b>2</b>
					<div>
						<strong>Add your notes (optional)</strong>
						<span>Upload or paste course notes so answers use them.</span>
					</div>
				</li>
				<li>
					<b>3</b>
					<div>
						<strong>Read and follow up</strong>
						<span>
							Ask a follow-up in the same conversation. Nothing leaves this
							device.
						</span>
					</div>
				</li>
			</ol>

			<div className="grid gap-5 lg:grid-cols-[1.15fr_.85fr]">
				<section
					id="study-assistant"
					className="glass-card lift student-chat-card rounded-[26px] p-5 sm:p-7"
					aria-labelledby="chat-heading"
				>
					<div className="field-row">
						<h2 id="chat-heading" className="section-title">
							Study assistant
						</h2>
						{history.length > 0 && (
							<button
								type="button"
								className="btn-quiet"
								disabled={!!busy}
								onClick={() => setHistory([])}
							>
								New conversation
							</button>
						)}
					</div>

					<div
						className="student-chat-thread"
						ref={threadRef}
						aria-live="polite"
					>
						{history.length === 0 && !showModelLoading ? (
							<div className="student-chat-empty">
								<p>Start with a question about your course. For example:</p>
								<div className="example-chips">
									{examples.map((ex) => (
										<button
											key={ex}
											type="button"
											className="example-chip"
											onClick={() => {
												setMode("ask");
												setQ(ex);
											}}
										>
											{ex}
										</button>
									))}
								</div>
							</div>
						) : (
							history.map((m, i) => (
								// Messages are append-only, so position is a stable identity here.
								// biome-ignore lint/suspicious/noArrayIndexKey: append-only list
								<article key={i} className={`student-message ${m.role}`}>
									<p className="student-message-label">
										{m.role === "user" ? "You" : "Assistant"}
									</p>
									<p className="student-message-content">{m.content}</p>
								</article>
							))
						)}
						{showModelLoading && <ChatLoading label={busy} progress={progress} />}
					</div>

					<div className="mt-6 grid gap-3">
						<div
							className="ui-tabs"
							role="tablist"
							aria-label="What do you want to do?"
						>
							<button
								type="button"
								role="tab"
								aria-selected={!isSummary}
								aria-controls="composer"
								className="ui-tab"
								onClick={() => setMode("ask")}
							>
								Ask a question
							</button>
							<button
								type="button"
								role="tab"
								aria-selected={isSummary}
								aria-controls="composer"
								className="ui-tab"
								onClick={() => setMode("summary")}
							>
								Summarize text
							</button>
						</div>
						<div id="composer" role="tabpanel">
							<label htmlFor="student-input">
								{isSummary ? "Text to summarize" : "Your question"}
							</label>
							<textarea
								id="student-input"
								className={`mt-1 ${isSummary ? "min-h-40" : "min-h-24"}`}
								value={q}
								onChange={(e) => setQ(e.target.value)}
								onKeyDown={onComposerKeyDown}
								placeholder={
									isSummary
										? "Paste a passage or your notes. To summarize an uploaded file, type “Summarize my notes”."
										: "e.g. Why is binary search faster than linear search?"
								}
							/>
						</div>
						<div className="field-row">
							<span className="kbd-hint">
								{isSummary ? (
									<>
										Press <kbd>Ctrl</kbd> + <kbd>Enter</kbd> to summarize
									</>
								) : (
									<>
										Press <kbd>Enter</kbd> to send · <kbd>Shift</kbd> +{" "}
										<kbd>Enter</kbd> for a new line
									</>
								)}
							</span>
							<button
								type="button"
								className="btn-primary"
								disabled={!!busy || !q.trim()}
								onClick={send}
							>
								{isSummary ? "Summarize" : "Send"} →
							</button>
						</div>
						{error && (
							<ErrorModal
								title={errorTitle(error)}
								message={error}
								onClose={() => setError("")}
							/>
						)}
					</div>
				</section>

				<section
					id="your-notes"
					className="glass-card lift rounded-[26px] p-5 sm:p-7"
					aria-labelledby="notes-heading"
				>
					<h2 id="notes-heading" className="section-title">
						Your notes <span className="optional-tag">Optional</span>
					</h2>
					<p className="section-copy">
						Add course material and the assistant will answer only from it. You
						can skip this and just ask. Remove your notes to go back to general
						answers.
					</p>
					<div className="mt-6 grid gap-4">
						<div className="grid gap-3">
							<p className="step-label">
								<b>A</b>Upload a file
							</p>
							<FilePicker
								file={file}
								onChange={setFile}
								disabled={!!busy}
								hint="PDF, TXT, MD, CSV, or a JPG/PNG photo of your notes"
							/>
							<button
								type="button"
								className="btn-secondary"
								disabled={!!busy || !file}
								onClick={upload}
							>
								Add file
							</button>
						</div>
						<div className="grid gap-3">
							<label htmlFor="student-notes" className="step-label">
								<b>B</b>Or paste text
							</label>
							<textarea
								id="student-notes"
								className="min-h-28"
								value={text}
								onChange={(e) => setText(e.target.value)}
								placeholder="Paste notes or a textbook section…"
							/>
							<button
								type="button"
								className="btn-secondary"
								disabled={!!busy || !text.trim()}
								onClick={index}
							>
								Add pasted notes
							</button>
						</div>
						{materials.length > 0 && (
							<div>
								<p className="step-label">Ready to use</p>
								<ul className="material-list">
									{materials.map((m) => (
										<li key={m.id}>{m.name}</li>
									))}
								</ul>
							</div>
						)}
						<button
							type="button"
							className="btn-quiet justify-self-start"
							disabled={!!busy}
							onClick={clearNotes}
						>
							Remove all notes
						</button>
					</div>
				</section>
			</div>

			{showIndexingOverlay && (
				<Loading
					label={busy}
					hint="Preparing your material for source-grounded answers."
					showGame={true}
					onCancel={cancelOperation}
					progress={progress}
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
