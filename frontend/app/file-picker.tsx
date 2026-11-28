"use client";

import { useRef } from "react";

const ACCEPT = ".pdf,.txt,.md,.csv,.png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp";

/** Upload area with a real "Choose a document" button that opens the file dialog. */
export default function FilePicker({
  file,
  onChange,
  hint,
  disabled = false,
}: {
  file: File | null;
  onChange: (file: File | null) => void;
  hint: string;
  disabled?: boolean;
}) {
  const input = useRef<HTMLInputElement>(null);
  return (
    <div className="file-picker">
      <button
        type="button"
        className="file-picker-button"
        onClick={() => input.current?.click()}
        disabled={disabled}
      >
        <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        {file ? "Change document" : "Choose a document"}
      </button>
      {file && (
        <span className="file-picker-name" title={file.name}>
          {file.name}
        </span>
      )}
      <span className="field-hint">{hint}</span>
      <input
        ref={input}
        className="sr-only"
        type="file"
        accept={ACCEPT}
        tabIndex={-1}
        aria-hidden="true"
        onChange={(e) => {
          onChange(e.target.files?.[0] || null);
          e.target.value = ""; // allow picking the same file again after removing it
        }}
      />
    </div>
  );
}
