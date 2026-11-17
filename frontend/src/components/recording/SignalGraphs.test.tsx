// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import SignalGraphs from "./SignalGraphs";
import { toRows } from "@/lib/signalGraph";
import { trimWindow, type SignalSample } from "@/hooks/useRecordingSignals";

let root: Root;
let container: HTMLDivElement;
beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  window.localStorage.clear();
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});
afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

const sample = (t: number, pan = t): SignalSample => ({ t, joints: { shoulder_pan: pan, gripper: 10 } });
const signals = { action: [sample(1), sample(2)], observation: [sample(1), sample(2)] };
const button = (label: string) =>
  Array.from(container.querySelectorAll("button")).find(
    (b) => b.getAttribute("aria-label") === label || b.textContent?.includes(label),
  )!;

describe("SignalGraphs", () => {
  it("shows an Action and an Observation graph", () => {
    act(() => root.render(<SignalGraphs signals={signals} />));
    expect(container.querySelector('[aria-label="Action graph"] svg.recharts-surface')).not.toBeNull();
    expect(container.querySelector('[aria-label="Observation graph"] svg.recharts-surface')).not.toBeNull();
  });

  it("waits for samples before drawing", () => {
    act(() => root.render(<SignalGraphs signals={{ action: [], observation: [] }} />));
    expect(container.textContent).toContain("Waiting for the arms…");
  });

  it("hides and remembers it", () => {
    act(() => root.render(<SignalGraphs signals={signals} />));
    act(() => button("Hide graphs").click());
    expect(container.querySelector('[aria-label="Action graph"]')).toBeNull();
    expect(window.localStorage.getItem("lelab.recordingGraphsHidden")).toBe("1");

    act(() => root.unmount());
    root = createRoot(container);
    act(() => root.render(<SignalGraphs signals={signals} />));
    expect(container.querySelector('[aria-label="Action graph"]')).toBeNull();
    act(() => button("Show graphs").click());
    expect(container.querySelector('[aria-label="Action graph"]')).not.toBeNull();
  });

  it("collapses one graph", () => {
    act(() => root.render(<SignalGraphs signals={signals} />));
    act(() => button("Action").click());
    expect(container.querySelector('[aria-label="Action graph"] svg.recharts-surface')).toBeNull();
    expect(container.querySelector('[aria-label="Observation graph"] svg.recharts-surface')).not.toBeNull();
  });
});

describe("signal helpers", () => {
  it("keeps only the last window of samples", () => {
    const samples = [0, 3, 6, 9, 12].map((t) => sample(t));
    expect(trimWindow(samples, 6).map((s) => s.t)).toEqual([6, 9, 12]);
    expect(trimWindow([], 6)).toEqual([]);
  });

  it("times rows from the newest sample and thins long histories", () => {
    expect(toRows([sample(8), sample(10)], 10)).toEqual([
      { t: -2, shoulder_pan: 8, gripper: 10 },
      { t: 0, shoulder_pan: 10, gripper: 10 },
    ]);
    const many = Array.from({ length: 450 }, (_, i) => sample(i / 30));
    const rows = toRows(many, many[449].t);
    expect(rows.length).toBe(150);
    expect(rows.at(-1)?.t).toBe(0);
  });
});
