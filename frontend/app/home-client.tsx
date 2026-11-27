"use client";

import { useEffect, useRef, useState } from "react";

/** Plays only while on screen (saves CPU for the local model) and never autoplays for reduced-motion users. */
export function LazyVideo({ src, label }: { src: string; label: string }) {
  const ref = useRef<HTMLVideoElement>(null);
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(media.matches);
    const el = ref.current;
    if (!el || media.matches) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) void el.play().catch(() => undefined);
        else el.pause();
      },
      { threshold: 0.25 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  return (
    <video
      ref={ref}
      src={src}
      loop
      muted
      playsInline
      preload="metadata"
      controls={reduced}
      aria-label={label}
    />
  );
}
