"use client";

import { useState } from "react";

/** Access-code field with an eye button to show or hide what was typed. */
export default function PasswordInput({
  id,
  value,
  onChange,
  placeholder = "Enter access code",
  autoFocus = false,
}: {
  id?: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  autoFocus?: boolean;
}) {
  const [visible, setVisible] = useState(false);
  return (
    <span className="password-field">
      <input
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        type={visible ? "text" : "password"}
        placeholder={placeholder}
        autoFocus={autoFocus}
        autoComplete="current-password"
        spellCheck={false}
      />
      <button
        type="button"
        className="password-toggle"
        onClick={() => setVisible((v) => !v)}
        aria-label={visible ? "Hide access code" : "Show access code"}
        aria-pressed={visible}
        title={visible ? "Hide access code" : "Show access code"}
      >
        {visible ? (
          <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
            <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
            <path d="M14.12 14.12a3 3 0 1 1-4.24-4.24" />
            <line x1="1" y1="1" x2="23" y2="23" />
          </svg>
        ) : (
          <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
            <circle cx="12" cy="12" r="3" />
          </svg>
        )}
      </button>
    </span>
  );
}
