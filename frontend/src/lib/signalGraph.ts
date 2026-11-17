import type { SignalSample } from "@/hooks/useRecordingSignals";

// Enough points for a smooth 10 s line without redrawing every frame's sample.
const MAX_POINTS = 150;

/** SO-101 joints in arm order, each with the same colour in both graphs. */
export const JOINTS: { key: string; label: string; color: string }[] = [
  { key: "shoulder_pan", label: "pan", color: "#f97316" },
  { key: "shoulder_lift", label: "lift", color: "#38bdf8" },
  { key: "elbow_flex", label: "elbow", color: "#4ade80" },
  { key: "wrist_flex", label: "flex", color: "#a78bfa" },
  { key: "wrist_roll", label: "roll", color: "#f472b6" },
  { key: "gripper", label: "grip", color: "#facc15" },
];

/**
 * Chart rows, `t` in seconds relative to `now` (so 0 is the newest sample),
 * thinned to at most MAX_POINTS by keeping every n-th sample counted back
 * from the newest, which therefore always stays.
 */
export function toRows(samples: SignalSample[], now: number): Record<string, number>[] {
  const step = Math.max(1, Math.ceil(samples.length / MAX_POINTS));
  const rows: Record<string, number>[] = [];
  for (let i = samples.length - 1; i >= 0; i -= step) {
    rows.push({ t: samples[i].t - now, ...samples[i].joints });
  }
  return rows.reverse();
}
