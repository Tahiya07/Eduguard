"use client";

import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { createPortal } from "react-dom";

export type HeroClip = { src: string; title: string; caption: string };

/**
 * Intro hero: copy + actions on the left, a featured video with selectable
 * thumbnails on the right. Uses the existing glass/blue palette only.
 */
export default function VideoHero({
  eyebrow,
  title,
  description,
  actions,
  trust = [],
  clips,
}: {
  eyebrow: string;
  title: ReactNode;
  description: ReactNode;
  actions?: ReactNode;
  trust?: string[];
  clips: HeroClip[];
}) {
  const [active, setActive] = useState(0);
  const [open, setOpen] = useState(false);
  const [reduceMotion, setReduceMotion] = useState(false);
  const clip = clips[active] ?? clips[0];

  useEffect(() => {
    setReduceMotion(window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }, []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <section className="video-hero glass-card rounded-[30px] p-5 sm:p-8">
      <div className="video-hero-copy">
        <p className="eyebrow">{eyebrow}</p>
        <h1 className="display">{title}</h1>
        <p>{description}</p>
        {actions && <div className="hero-actions">{actions}</div>}
        {trust.length > 0 && (
          <div className="hero-trust">
            {trust.map((t) => (
              <span key={t}>{t}</span>
            ))}
          </div>
        )}
      </div>

      <div className="hero-stage">
        <div className="hero-stage-main">
          <video
            key={clip.src}
            src={clip.src}
            autoPlay={!reduceMotion}
            loop
            muted
            playsInline
            preload="auto"
            aria-label={clip.title}
          />
          <div className="hero-stage-caption">
            <div>
              <h3>{clip.title}</h3>
              <p>{clip.caption}</p>
            </div>
            <button type="button" onClick={() => setOpen(true)}>
              Watch full ▸
            </button>
          </div>
        </div>
        {clips.length > 1 && (
          <div className="hero-thumbs" role="group" aria-label="Choose a video">
            {clips.map((c, i) => (
              <button
                key={c.src}
                type="button"
                className="hero-thumb"
                aria-pressed={i === active}
                aria-label={`Show video: ${c.title}`}
                onClick={() => setActive(i)}
              >
                {/* #t= shows a real frame instead of a blank poster */}
                <video src={`${c.src}#t=0.8`} muted playsInline preload="metadata" tabIndex={-1} />
                <span>{c.title}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {open && createPortal(
        <div
          className="reel-dialog-backdrop"
          role="dialog"
          aria-modal="true"
          aria-label={clip.title}
          onClick={() => setOpen(false)}
        >
          <section className="reel-dialog" onClick={(e) => e.stopPropagation()}>
            <button type="button" className="reel-close" onClick={() => setOpen(false)} aria-label="Close video">
              ×
            </button>
            <video src={clip.src} controls autoPlay playsInline />
            <div>
              <p className="eyebrow">{eyebrow}</p>
              <h2 className="section-title">{clip.title}</h2>
              <p className="section-copy">{clip.caption}</p>
            </div>
          </section>
        </div>,
        document.body,
      )}
    </section>
  );
}
