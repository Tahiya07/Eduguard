"use client";

import { useEffect, useState } from "react";

export const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Role = "student" | "teacher";
export type Session = { token: string; role: Role };

const KEY = "eduguard.session";
const EVENT = "eduguard-session";

// The session lives in sessionStorage (this tab only, cleared when the tab closes), so moving
// between Home and a workspace keeps the same server workspace and its uploaded notes.
export function readSession(): Session | null {
  try {
    const raw = window.sessionStorage.getItem(KEY);
    if (!raw) return null;
    const s = JSON.parse(raw) as Session;
    return s?.token && (s.role === "student" || s.role === "teacher") ? s : null;
  } catch {
    return null;
  }
}

export function saveSession(session: Session) {
  try {
    window.sessionStorage.setItem(KEY, JSON.stringify(session));
  } catch {
    // storage blocked: the session simply stays in memory for this page
  }
  window.dispatchEvent(new Event(EVENT));
}

export function clearSession() {
  try {
    window.sessionStorage.removeItem(KEY);
  } catch {
    // ignore
  }
  window.dispatchEvent(new Event(EVENT));
}

export function useSession(): Session | null {
  const [session, setSession] = useState<Session | null>(null);
  useEffect(() => {
    const sync = () => setSession(readSession());
    sync();
    window.addEventListener(EVENT, sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(EVENT, sync);
      window.removeEventListener("storage", sync);
    };
  }, []);
  return session;
}
