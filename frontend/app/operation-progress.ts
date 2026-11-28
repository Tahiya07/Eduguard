"use client";

import { useEffect, useState } from "react";

/** What a running backend operation is doing right now, e.g. "Summarizing part 3 of 9: Homeostasis…". */
export type OperationProgress = {
  message: string | null;
  step?: number | null;
  total?: number | null;
};

/**
 * Polls GET /operations/progress once a second while `active`, so the waiting dialog can explain
 * why a step takes time (section-by-section summaries, rewrite retries, OCR of scanned pages).
 * `getId` must be stable (wrap it in useCallback); it returns the running operation's id.
 */
export function useOperationProgress(
  api: string,
  token: string | null | undefined,
  getId: () => string | undefined,
  active: boolean,
): OperationProgress | null {
  const [progress, setProgress] = useState<OperationProgress | null>(null);

  useEffect(() => {
    if (!active || !token) {
      setProgress(null);
      return;
    }
    let stopped = false;
    const tick = async () => {
      const id = getId();
      if (!id) return;
      try {
        const r = await fetch(`${api}/operations/progress?operation_id=${encodeURIComponent(id)}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!r.ok) return;
        const d = (await r.json()) as OperationProgress;
        if (!stopped && d && d.message) setProgress(d);
      } catch {
        // Progress is informational; a failed poll is simply skipped.
      }
    };
    void tick();
    const timer = window.setInterval(tick, 1000);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [api, token, getId, active]);

  return progress;
}
