// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import Inference from "./Inference";
import { button, click, json, type } from "@/components/remote/testUtils";
import { InteractiveCommand, summarizeCommands } from "@/lib/inferenceApi";

const mocks = vi.hoisted(() => ({ fetcher: vi.fn(), navigate: vi.fn(), toast: vi.fn() }));
vi.mock("@/contexts/ApiContext", () => ({ useApi: () => ({ baseUrl: "http://test", fetchWithHeaders: mocks.fetcher }) }));
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ toast: mocks.toast }) }));
vi.mock("react-router-dom", () => ({ useNavigate: () => mocks.navigate }));

const command = (id: number, status: InteractiveCommand["status"], how: string | null, elapsed = 10): InteractiveCommand => ({
  id, text: `command ${id}`, status, how, submitted_at: 100 + id, started_at: status === "pending" ? null : 101 + id,
  ended_at: status === "running" || status === "pending" ? null : 120 + id, elapsed_s: elapsed,
});
const FINISHED = [
  command(1, "completed", "arm settled", 23),
  command(2, "not_completed", "time limit", 60),
  command(3, "stopped", "stopped", 5),
  command(4, "completed", "marked done", 12),
];
const session = (commands: InteractiveCommand[], state = "ready", active = true) => ({
  inference_active: active, mode: "remote", started_at: 1, rollout_started_at: 2, elapsed_s: 30, rollout_elapsed_s: 28,
  duration_s: null, policy_ref: "smolvla @ 10.2.0.7:5555", log_path: null, video: null, arm_link_issues: [],
  interactive: {
    state, command_time_s: 60, return_between_commands: true, auto_complete: true,
    chunk_blending: false, blend_steps: 10, commands,
  },
});

let status: Record<string, unknown>;
let calls: { path: string; body: unknown }[];
let root: Root;
let container: HTMLDivElement;

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  calls = [];
  mocks.fetcher.mockReset(); mocks.navigate.mockReset(); mocks.toast.mockReset();
  mocks.fetcher.mockImplementation(async (url: string, init: RequestInit = {}) => {
    const path = url.replace("http://test", "");
    if (path === "/inference-status") return json(status);
    calls.push({ path, body: init.body ? JSON.parse(String(init.body)) : null });
    return json({ success: true, message: "ok" });
  });
});
afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

const flush = () => act(async () => { await new Promise((r) => setTimeout(r, 0)); });
const render = async () => {
  await act(async () => root.render(<Inference />));
  await flush();
};
const rows = () => Array.from(container.querySelectorAll<HTMLElement>('[data-testid="command-row"]'));
const input = () => container.querySelector<HTMLInputElement>('input[aria-label="Command"]')!;

test("lists every command with how it ended, and Enter sends the next one", async () => {
  status = session(FINISHED);
  await render();

  expect(rows().map((row) => row.dataset.status)).toEqual(["completed", "not_completed", "stopped", "completed"]);
  expect(container.textContent).toContain("the arm stopped moving");
  expect(container.textContent).toContain("Didn't complete · not done within 1:00");
  expect(container.textContent).toContain("stopped by you");
  expect(container.textContent).toContain("READY");
  expect(button(container, "End session")).toBeTruthy();

  type(input(), "stack the cubes");
  await act(async () => {
    input().dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  });
  await flush();
  expect(calls).toContainEqual({ path: "/inference-command", body: { text: "stack the cubes" } });
  expect(input().value).toBe("");
});

test("a running command can be marked done or stopped, and blocks the next one", async () => {
  status = session([...FINISHED, command(5, "running", null, 12)], "running");
  await render();

  expect(rows().at(-1)!.dataset.status).toBe("running");
  expect(container.textContent).toContain("Running “command 5”");
  type(input(), "next");
  expect(container.querySelector<HTMLButtonElement>('button[aria-label="Send command"]')!.disabled).toBe(true);
  expect(container.textContent).toContain("A command is running");

  await click(button(container, "Done"));
  await click(button(container, "Stop command"));
  expect(calls.filter((c) => c.path === "/inference-command/end").map((c) => c.body)).toEqual([
    { completed: true },
    { completed: false },
  ]);
  expect(calls.some((c) => c.path === "/inference-command")).toBe(false);
});

test("a refused command shows why", async () => {
  status = session([]);
  mocks.fetcher.mockImplementation(async (url: string) =>
    url.endsWith("/inference-status")
      ? json(status)
      : json({ detail: "The arm is going back to its start pose. Send the command when it's there." }, 409),
  );
  await render();
  expect(container.textContent).toContain("No commands yet");
  type(input(), "pick");
  await click(container.querySelector<HTMLButtonElement>('button[aria-label="Send command"]')!);
  await flush();
  expect(mocks.toast).toHaveBeenCalledWith(
    expect.objectContaining({ description: "The arm is going back to its start pose. Send the command when it's there." }),
  );
  expect(input().value).toBe("pick"); // kept, to send again
});

test("an ended session stays on the page with its commands", async () => {
  status = { ...session(FINISHED.slice(0, 2), "ready", false), exited: true, exit_code: 0, outcome: "ok" };
  await render();

  expect(mocks.navigate).not.toHaveBeenCalled();
  expect(mocks.toast).toHaveBeenCalledWith(
    expect.objectContaining({ title: "Session ended", description: "2 commands: 1 done, 1 didn't complete." }),
  );
  expect(rows()).toHaveLength(2);
  expect(container.textContent).toContain("Back to home");
  expect(container.querySelector('input[aria-label="Command"]')).toBeNull();
});

const blendBox = () => container.querySelector<HTMLButtonElement>("#session-chunk-blending")!;

test("chunk blending switches between commands", async () => {
  status = session(FINISHED);
  await render();
  expect(blendBox().disabled).toBe(false);
  expect(blendBox().getAttribute("data-state")).toBe("unchecked");
  expect(container.textContent).toContain("Applies from the next command");

  await click(blendBox());
  await flush();
  expect(calls).toContainEqual({ path: "/inference-settings", body: { chunk_blending: true } });
  expect(blendBox().getAttribute("data-state")).toBe("checked"); // shown at once, before the next status
});

test("chunk blending is locked while a command runs", async () => {
  status = session([command(1, "running", null, 3)], "running");
  await render();
  expect(blendBox().disabled).toBe(true);
  expect(container.textContent).toContain("Change it between commands.");
});

test("a plain run shows the blending it started with", async () => {
  status = {
    inference_active: true, mode: "remote", started_at: 1, rollout_started_at: 2, elapsed_s: 9, rollout_elapsed_s: 7,
    duration_s: 60, policy_ref: "act @ 10.2.0.7:5555", log_path: null, video: null, arm_link_issues: [], interactive: null,
    remote: {
      address: "10.2.0.7:5555", policy: "act", phase: "running", steps: 200, fps: 30, chunks: 12, rtt_ms_median: 40,
      infer_ms_median: 20, wait_s: 0, actions_per_chunk: 50, arm_skipped: 0, blend_steps: 8, blended: 88,
    },
  };
  await render();
  expect(container.textContent).toContain("blending over 8 actions (88 blended)");
  expect(container.querySelector("#session-chunk-blending")).toBeNull();
});

test("summarizes a session's commands", () => {
  expect(summarizeCommands([])).toBe("No commands were run.");
  expect(summarizeCommands([command(1, "completed", "arm settled")])).toBe("1 command: 1 done.");
  expect(summarizeCommands([...FINISHED, command(5, "failed", "boom")])).toBe(
    "5 commands: 2 done, 1 didn't complete, 1 stopped, 1 failed.",
  );
});
