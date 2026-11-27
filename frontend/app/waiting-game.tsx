"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";

/* Two short games for the wait while the local model works:
   1. Guess the level: pick the Bloom level a sample question tests.
   2. Word scramble: unscramble an academic term from its letters and a hint.
   Scores and the chosen game persist in this browser; everything is built in (works offline). */

type Level = "Remember" | "Understand" | "Apply" | "Analyze" | "Evaluate" | "Create";
const LEVELS: Level[] = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"];

const BLOOM_ITEMS: { q: string; level: Level; why: string }[] = [
  // Remember
  { q: "List the seven layers of the OSI model.", level: "Remember", why: "Recalling a known list." },
  { q: "What does the acronym SQL stand for?", level: "Remember", why: "Retrieving a single fact from memory." },
  { q: "Name the four conditions necessary for a deadlock.", level: "Remember", why: "Recalling terms, not explaining them." },
  { q: "State Ohm's law.", level: "Remember", why: "Reproducing a law as learned." },
  { q: "Define the term 'opportunity cost'.", level: "Remember", why: "Giving a definition from memory." },
  { q: "What is the SI unit of stress?", level: "Remember", why: "Recalling a fact." },
  // Understand
  { q: "Explain why binary search needs a sorted array.", level: "Understand", why: "Explaining a reason in your own words." },
  { q: "Describe how a hash table resolves collisions using chaining.", level: "Understand", why: "Describing a mechanism in your own words." },
  { q: "Explain the difference between a process and a thread.", level: "Understand", why: "Clarifying two related concepts." },
  { q: "Explain why a transformer does not work on a DC supply.", level: "Understand", why: "Explaining a cause." },
  { q: "Summarize the law of supply and demand.", level: "Understand", why: "Summarizing shows comprehension." },
  { q: "Interpret what the slope of a stress–strain curve represents.", level: "Understand", why: "Interpreting meaning from a graph." },
  // Apply
  { q: "Write a SQL query that returns all students with a CGPA above 3.5.", level: "Apply", why: "Using a technique on a given task." },
  { q: "Convert the binary number 101101 to decimal.", level: "Apply", why: "Executing a known procedure." },
  { q: "Show the array [5, 2, 8, 1] after the first pass of bubble sort.", level: "Apply", why: "Carrying out an algorithm on given data." },
  { q: "Calculate the current through a 10 Ω resistor across a 5 V source.", level: "Apply", why: "Using a formula in a concrete case." },
  { q: "Evaluate the integral of 3x² from 0 to 2.", level: "Apply", why: "Applying a standard method (despite the verb)." },
  { q: "Find the bending moment at the centre of a 6 m simply supported beam with a 10 kN central load.", level: "Apply", why: "Applying a formula to new values." },
  // Analyze
  { q: "Compare how TCP and UDP handle lost packets.", level: "Analyze", why: "Comparing structures and their differences." },
  { q: "Analyze why this recursive Fibonacci function runs in exponential time.", level: "Analyze", why: "Breaking behaviour into its causes." },
  { q: "Distinguish between the causes of a race condition and a deadlock in this code.", level: "Analyze", why: "Differentiating related problems." },
  { q: "Distinguish between correlation and causation in this survey's results.", level: "Analyze", why: "Separating relevant from irrelevant relations." },
  { q: "Identify the assumptions behind the author's argument.", level: "Analyze", why: "Uncovering underlying structure." },
  { q: "Examine how a rise in interest rates affects consumer spending and investment.", level: "Analyze", why: "Tracing relationships between parts." },
  // Evaluate
  { q: "Judge whether a NoSQL database suits a banking system better than a relational one, and justify your view.", level: "Evaluate", why: "Making and defending a judgment." },
  { q: "Assess whether insertion sort is a good choice for nearly sorted data.", level: "Evaluate", why: "Judging against criteria." },
  { q: "Verify whether the proposed algorithm always terminates.", level: "Evaluate", why: "Checking a claim for correctness." },
  { q: "Critique this lab experiment's design and recommend one improvement.", level: "Evaluate", why: "Critiquing with justification." },
  { q: "Which of the two bridge designs offers better safety for its cost? Defend your answer.", level: "Evaluate", why: "Choosing and justifying between options." },
  { q: "Evaluate whether the contract clause is enforceable on the stated facts.", level: "Evaluate", why: "Reaching a reasoned judgment." },
  // Create
  { q: "Design a database schema for a university course-registration system.", level: "Create", why: "Planning something new." },
  { q: "Develop an algorithm that finds duplicate files in a folder.", level: "Create", why: "Producing an original solution." },
  { q: "Propose a caching strategy to reduce the load on a web server.", level: "Create", why: "Generating a new plan." },
  { q: "Design a circuit that switches on a fan when the room exceeds 30 °C.", level: "Create", why: "Putting parts together into a new whole." },
  { q: "Propose a marketing plan for a campus startup.", level: "Create", why: "Creating an original strategy." },
  { q: "Compose a short story that illustrates dramatic irony.", level: "Create", why: "Creating an original work." },
];

const WORDS: { word: string; hint: string }[] = [
  // CSE
  { word: "ALGORITHM", hint: "Step-by-step procedure" },
  { word: "RECURSION", hint: "A function calling itself" },
  { word: "COMPILER", hint: "Turns source code into machine code" },
  { word: "DATABASE", hint: "Organized store of records" },
  { word: "NETWORK", hint: "Connected computers or nodes" },
  { word: "SYNTAX", hint: "Rules of a language's structure" },
  { word: "VARIABLE", hint: "Named value that can change" },
  { word: "BOOLEAN", hint: "True-or-false data type" },
  { word: "PROTOCOL", hint: "Agreed rules for communication" },
  { word: "ENCRYPTION", hint: "Scrambling data so only keys can read it" },
  { word: "DEADLOCK", hint: "Processes waiting on each other forever" },
  { word: "CACHE", hint: "Small, fast memory for recent data" },
  { word: "KERNEL", hint: "Core of an operating system" },
  { word: "BANDWIDTH", hint: "Data rate of a link" },
  { word: "POINTER", hint: "Variable holding a memory address" },
  { word: "FIREWALL", hint: "Filters network traffic" },
  { word: "QUEUE", hint: "First in, first out" },
  { word: "STACK", hint: "Last in, first out" },
  // Other departments
  { word: "VOLTAGE", hint: "Electric potential difference (EEE)" },
  { word: "CAPACITOR", hint: "Stores electric charge (EEE)" },
  { word: "TRANSISTOR", hint: "Semiconductor switch or amplifier (EEE)" },
  { word: "CONCRETE", hint: "Cement, sand, aggregate and water (Civil)" },
  { word: "TORQUE", hint: "Rotational force (Mechanical)" },
  { word: "FRICTION", hint: "Force resisting sliding (Mechanical)" },
  { word: "MOMENTUM", hint: "Mass times velocity (Physics)" },
  { word: "ENTROPY", hint: "Measure of disorder (Physics)" },
  { word: "DERIVATIVE", hint: "Instantaneous rate of change (Mathematics)" },
  { word: "INTEGRAL", hint: "Area under a curve (Mathematics)" },
  { word: "MATRIX", hint: "Rectangular array of numbers (Mathematics)" },
  { word: "THEOREM", hint: "Statement proven true (Mathematics)" },
  { word: "INFLATION", hint: "Rising general price level (Economics)" },
  { word: "MARKET", hint: "Where buyers and sellers meet (Business)" },
  { word: "DEPRECIATION", hint: "Loss of an asset's value over time (Accounting)" },
  { word: "CONTRACT", hint: "Legally binding agreement (Law)" },
  { word: "METAPHOR", hint: "Figure of speech comparing two things (English)" },
  { word: "ENZYME", hint: "Biological catalyst (Biology)" },
];

const STORE = "eduguard.waitingGame";
type Saved = { tab: "bloom" | "scramble"; bloomScore: number; bloomTotal: number; best: number; words: number; cursor: number };
const DEFAULTS: Saved = { tab: "bloom", bloomScore: 0, bloomTotal: 0, best: 0, words: 0, cursor: 0 };

function load(): Saved {
  try {
    return { ...DEFAULTS, ...JSON.parse(window.localStorage.getItem(STORE) || "{}") };
  } catch {
    return DEFAULTS;
  }
}

function persist(s: Saved) {
  try {
    window.localStorage.setItem(STORE, JSON.stringify(s));
  } catch {
    // storage unavailable: scores just reset next time
  }
}

function shuffled<T>(items: T[]): T[] {
  const a = [...items];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function scramble(word: string): string {
  let out = word;
  for (let tries = 0; tries < 8 && out === word; tries++) out = shuffled(word.split("")).join("");
  return out;
}

function BloomQuiz({ saved, update }: { saved: Saved; update: (p: Partial<Saved>) => void }) {
  const order = useMemo(() => shuffled(BLOOM_ITEMS.map((_, i) => i)), []);
  const [pos, setPos] = useState(saved.cursor % BLOOM_ITEMS.length);
  const [picked, setPicked] = useState<Level | null>(null);
  const [streak, setStreak] = useState(0);
  const item = BLOOM_ITEMS[order[pos % order.length]];

  const choose = (level: Level) => {
    if (picked) return;
    setPicked(level);
    const right = level === item.level;
    const nextStreak = right ? streak + 1 : 0;
    setStreak(nextStreak);
    update({
      bloomScore: saved.bloomScore + (right ? 1 : 0),
      bloomTotal: saved.bloomTotal + 1,
      best: Math.max(saved.best, nextStreak),
    });
  };
  const next = () => {
    setPicked(null);
    setPos((p) => p + 1);
    update({ cursor: saved.cursor + 1 });
  };

  return (
    <div className="wg-panel">
      <div className="waiting-game-stats wg-stats">
        <span>Score {saved.bloomScore}/{saved.bloomTotal}</span>
        {streak > 1 && <span>· Streak {streak}</span>}
        {saved.best > 1 && <span>· Best {saved.best}</span>}
      </div>
      <p className="wg-question">“{item.q}”</p>
      <div className="wg-levels" role="group" aria-label="Choose the Bloom level">
        {LEVELS.map((level) => {
          const state = !picked ? "" : level === item.level ? "right" : level === picked ? "wrong" : "dim";
          return (
            <button key={level} type="button" className={`wg-level ${state}`} onClick={() => choose(level)} disabled={!!picked}>
              {level}
            </button>
          );
        })}
      </div>
      <div className="wg-feedback" aria-live="polite">
        {picked && (
          <>
            <p>
              <b>{picked === item.level ? "Correct!" : `It's ${item.level}.`}</b> {item.why}
            </p>
            <button type="button" className="btn-secondary wg-next" onClick={next} autoFocus>
              Next question →
            </button>
          </>
        )}
      </div>
    </div>
  );
}

function WordScramble({ saved, update }: { saved: Saved; update: (p: Partial<Saved>) => void }) {
  const order = useMemo(() => shuffled(WORDS.map((_, i) => i)), []);
  const [pos, setPos] = useState(0);
  const item = WORDS[order[pos % order.length]];
  const letters = useMemo(() => scramble(item.word), [item.word]);
  const [guess, setGuess] = useState("");
  const [hinted, setHinted] = useState(false);
  const [result, setResult] = useState<"" | "right" | "wrong" | "skipped">("");
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    input.current?.focus();
  }, [pos]);

  const next = () => {
    setPos((p) => p + 1);
    setGuess("");
    setHinted(false);
    setResult("");
  };
  const check = (e: FormEvent) => {
    e.preventDefault();
    if (result === "right") return next();
    if (guess.trim().toUpperCase() === item.word) {
      setResult("right");
      update({ words: saved.words + 1 });
    } else {
      setResult("wrong");
    }
  };

  return (
    <div className="wg-panel">
      <div className="waiting-game-stats wg-stats">
        <span>Words solved {saved.words}</span>
      </div>
      <div className="wg-tiles" aria-label={`Scrambled letters: ${letters.split("").join(" ")}`}>
        {letters.split("").map((ch, i) => (
          // letters can repeat, so position is part of the identity
          // biome-ignore lint/suspicious/noArrayIndexKey: static per word
          <span key={`${ch}-${i}`} className="wg-tile">{ch}</span>
        ))}
      </div>
      <p className="wg-hint">
        Hint: {item.hint}
        {hinted && <> · starts with <b>{item.word[0]}</b></>}
      </p>
      <form className="wg-form" onSubmit={check}>
        <label htmlFor="wg-guess" className="sr-only">Your answer</label>
        <input
          id="wg-guess"
          ref={input}
          value={guess}
          onChange={(e) => {
            setGuess(e.target.value);
            if (result === "wrong") setResult("");
          }}
          placeholder="Type the word"
          autoComplete="off"
          spellCheck={false}
          disabled={result === "right" || result === "skipped"}
        />
        <button type="submit" className="btn-primary" disabled={!guess.trim() && result !== "right"}>
          {result === "right" ? "Next" : "Check"}
        </button>
      </form>
      <div className="wg-feedback" aria-live="polite">
        {result === "right" && <p><b>Correct!</b> {item.word} — {item.hint.toLowerCase()}.</p>}
        {result === "wrong" && <p>Not quite, try again.</p>}
        {result === "skipped" && <p>It was <b>{item.word}</b>.</p>}
        <div className="wg-actions">
          {!hinted && result !== "right" && result !== "skipped" && (
            <button type="button" className="btn-quiet" onClick={() => setHinted(true)}>Show first letter</button>
          )}
          {result === "skipped" ? (
            <button type="button" className="btn-secondary" onClick={next} autoFocus>Next word →</button>
          ) : (
            result !== "right" && <button type="button" className="btn-quiet" onClick={() => setResult("skipped")}>Reveal</button>
          )}
        </div>
      </div>
    </div>
  );
}

const GAMES = [
  { id: "bloom" as const, title: "Guess the level", copy: "Read a question, pick its Bloom level.", meta: "36 questions · ~5 s a round" },
  { id: "scramble" as const, title: "Word scramble", copy: "Rebuild an academic term from a hint.", meta: "36 terms · type to solve" },
];

/** Mini Bloom staircase: six steps, lit one after another on hover. */
function BloomPreview() {
  return (
    <div className="wg-preview wg-preview-bloom" aria-hidden="true">
      {["R", "U", "Ap", "An", "E", "C"].map((label, i) => (
        <span key={label} className="wg-step" style={{ height: `${28 + i * 12}%`, animationDelay: `${i * 90}ms` }}>
          {label}
        </span>
      ))}
    </div>
  );
}

/** Letter tiles that rotate through scrambled words while the card is visible. */
function ScramblePreview() {
  const samples = ["HCACE", "CACHE", "KCATS", "STACK", "NELREK", "KERNEL"];
  const [i, setI] = useState(0);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const id = window.setInterval(() => setI((n) => (n + 1) % samples.length), 1100);
    return () => window.clearInterval(id);
  }, [samples.length]);
  const word = samples[i];
  const solved = i % 2 === 1;
  return (
    <div className={`wg-preview wg-preview-scramble ${solved ? "solved" : ""}`} aria-hidden="true">
      {word.split("").map((ch, k) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: decorative tiles
        <span key={`${i}-${k}`} className="wg-mini-tile" style={{ animationDelay: `${k * 45}ms` }}>
          {ch}
        </span>
      ))}
    </div>
  );
}

export default function WaitingGame({ active }: { active: boolean }) {
  const [saved, setSaved] = useState<Saved>(DEFAULTS);
  const [ready, setReady] = useState(false);
  // Each wait starts with the choice; the last game played is marked.
  const [playing, setPlaying] = useState<Saved["tab"] | null>(null);

  useEffect(() => {
    setSaved(load());
    setReady(true);
  }, []);

  const update = (patch: Partial<Saved>) =>
    setSaved((prev) => {
      const next = { ...prev, ...patch };
      persist(next);
      return next;
    });

  const start = (id: Saved["tab"]) => {
    setPlaying(id);
    update({ tab: id });
  };

  if (!active || !ready) return null;

  return (
    <div className="waiting-game-container">
      <div className="waiting-game-header">
        <span className="waiting-game-title">
          {playing ? GAMES.find((g) => g.id === playing)?.title : "Play while you wait"}
        </span>
        {playing && (
          <button type="button" className="btn-quiet wg-switch" onClick={() => start(playing === "bloom" ? "scramble" : "bloom")}>
            Switch to {playing === "bloom" ? "Word scramble" : "Guess the level"}
          </button>
        )}
      </div>
      {!playing ? (
        <div className="wg-choices" role="group" aria-label="Choose a game">
          {GAMES.map((g) => {
            const played = g.id === "bloom" ? saved.bloomTotal > 0 : saved.words > 0;
            const record = g.id === "bloom"
              ? played ? `Your score ${saved.bloomScore}/${saved.bloomTotal}${saved.best > 1 ? ` · best streak ${saved.best}` : ""}` : "New — try it"
              : played ? `${saved.words} word${saved.words === 1 ? "" : "s"} solved` : "New — try it";
            const last = saved.tab === g.id && (saved.bloomTotal > 0 || saved.words > 0);
            return (
              <button key={g.id} type="button" className={`wg-choice wg-choice-${g.id}`} onClick={() => start(g.id)}>
                {last && <em className="wg-badge">Last played</em>}
                {g.id === "bloom" ? <BloomPreview /> : <ScramblePreview />}
                <strong>{g.title}</strong>
                <span>{g.copy}</span>
                <small className="wg-meta">{g.meta}</small>
                <span className="wg-choice-foot">
                  <b>{record}</b>
                  <i>Play →</i>
                </span>
              </button>
            );
          })}
        </div>
      ) : playing === "bloom" ? (
        <BloomQuiz saved={saved} update={update} />
      ) : (
        <WordScramble saved={saved} update={update} />
      )}
    </div>
  );
}
