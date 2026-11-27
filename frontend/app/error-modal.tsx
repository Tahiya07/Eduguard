"use client";

import { useEffect, useRef } from "react";
import type { ReactNode } from "react";
import { createPortal } from "react-dom";

/** Centered warning dialog used for every error message in the workspaces. */
export default function ErrorModal({
  message,
  onClose,
  title = "Something went wrong",
  action,
}: {
  message: string;
  onClose: () => void;
  title?: string;
  action?: ReactNode;
}) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  // Runs once per opening: focus OK, close on Escape, give focus back afterwards.
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onCloseRef.current();
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      previous?.focus?.();
    };
  }, []);

  return createPortal(
    <div className="error-modal-backdrop" onClick={onClose}>
      <section
        className="error-modal glass-card"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="error-modal-title"
        aria-describedby="error-modal-message"
        onClick={(e) => e.stopPropagation()}
      >
        <span className="error-modal-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 9v4M12 17h.01" />
            <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" />
          </svg>
        </span>
        <h2 id="error-modal-title">{title}</h2>
        <p id="error-modal-message">{message}</p>
        <div className="error-modal-actions">
          {action}
          <button ref={closeRef} type="button" className="btn-primary" onClick={onClose}>
            OK
          </button>
        </div>
      </section>
    </div>,
    document.body,
  );
}

/** Short, specific title for common errors so the dialog reads clearly. */
export function errorTitle(message: string): string {
  const m = message.toLowerCase();
  if (m.includes("ocr") || m.includes("upload") || m.includes("file")) return "Couldn't add this file";
  if (m.includes("access code") || m.includes("session") || m.includes("credentials")) return "Couldn't sign in";
  if (m.includes("classifier")) return "Bloom classifier not ready";
  if (m.includes("failed to fetch") || m.includes("networkerror")) return "Can't reach the local service";
  return "Something went wrong";
}
