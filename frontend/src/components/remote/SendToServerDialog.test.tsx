// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import SendToServerDialog from "./SendToServerDialog";
import { RemoteServerProvider } from "@/contexts/RemoteServerContext";
import { button, click, json } from "./testUtils";

const mocks = vi.hoisted(() => ({ fetcher: vi.fn() }));
vi.mock("@/contexts/ApiContext", () => ({ useApi: () => ({ baseUrl: "http://test", fetchWithHeaders: mocks.fetcher }) }));
vi.mock("@/components/ui/select", async () => (await import("./testUtils")).uiMocks.select());
vi.mock("@/components/ui/dialog", async () => (await import("./testUtils")).uiMocks.dialog());

const SERVER = {
  name: "cluster", host: "10.2.100.14", port: 22, username: "student", auth: "password", key_path: "",
  remote_dataset_dir: "projects/data", remote_work_dir: "", session: { connected: true, home: "/home/student" },
};
const REPO = "local/pick_cube_20261117_120000";
let statuses: Record<string, unknown>[];
let servers: Record<string, unknown>[];
let uploadResponse: (() => Response) | null;
let calls: { method: string; path: string; body: Record<string, unknown> | null }[];
let root: Root;
let container: HTMLDivElement;

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  container = document.createElement("div");
  root = createRoot(container);
  calls = [];
  statuses = [{ state: "idle" }];
  servers = [SERVER];
  uploadResponse = null;
  window.localStorage.clear();
  mocks.fetcher.mockReset();
  mocks.fetcher.mockImplementation(async (url: string, init: RequestInit = {}) => {
    const method = init.method ?? "GET";
    const path = url.replace("http://test", "");
    const body = init.body ? JSON.parse(String(init.body)) : null;
    calls.push({ method, path, body });
    if (path === "/remote-servers") return json({ servers });
    if (path === "/remote-upload/status") return json(statuses.length > 1 ? statuses.shift() : statuses[0]);
    if (path === "/remote-upload") {
      if (uploadResponse) return uploadResponse();
      statuses = [
        { state: "running", dataset_repo_id: REPO, bytes_total: 200, bytes_done: 100, files_total: 4, files_done: 2, current_file: "videos/a.mp4" },
        { state: "done", dataset_repo_id: REPO, bytes_total: 200, bytes_done: 200, files_total: 4, files_done: 4, remote_path: "/home/student/projects/data/pick_cube_20261117_120000", train_command: "lerobot-train --dataset.root=/home/student/projects/data/pick_cube_20261117_120000" },
      ];
      return json({ state: "running", dataset_repo_id: REPO, bytes_total: 200, bytes_done: 0, files_total: 4, files_done: 0 });
    }
    if (path === "/remote-upload/cancel") return json({ success: true });
    return json({}, 404);
  });
});
afterEach(() => {
  act(() => root.unmount());
  vi.useRealTimers();
});

async function render(autoStart = false) {
  await act(async () =>
    root.render(
      <RemoteServerProvider>
        <SendToServerDialog open onOpenChange={vi.fn()} datasetRepoId={REPO} autoStart={autoStart} />
      </RemoteServerProvider>,
    ),
  );
}

test("sends the dataset to the profile's folder and shows progress until done", async () => {
  vi.useFakeTimers();
  await render();
  await click(button(container, "Send"));

  const post = calls.find((c) => c.path === "/remote-upload")!;
  expect(post.body).toEqual({
    server: "cluster", dataset_repo_id: REPO, remote_dir: "projects/data", remote_name: "pick_cube_20261117_120000",
    send_training_kit: true,
  });

  await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
  expect(container.textContent).toContain("50%");
  expect(container.textContent).toContain("videos/a.mp4");

  await click(button(container, "Cancel upload"));
  expect(calls.some((c) => c.path === "/remote-upload/cancel" && c.method === "POST")).toBe(true);

  await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
  expect(container.textContent).toContain("On the server at");
  expect(container.textContent).toContain("lerobot-train --dataset.root=");
});

test("autoStart sends once the server is connected", async () => {
  await render(true);
  await act(async () => { await Promise.resolve(); });
  expect(calls.filter((c) => c.path === "/remote-upload")).toHaveLength(1);
});

test("a session LeLab lost shows the Connect form again instead of a stale Connected", async () => {
  await render();
  expect(container.textContent).toContain("Connected to student@10.2.100.14");

  // LeLab restarted (lelab --dev reloads): the server has no session any more.
  servers = [{ ...SERVER, session: { connected: false } }];
  uploadResponse = () =>
    json({ detail: { message: "The connection to cluster is not open (LeLab may have restarted).", code: "not_connected" } }, 409);
  await click(button(container, "Send"));
  await act(async () => { await Promise.resolve(); });

  expect(container.textContent).toContain("is not open");
  expect(container.textContent).toContain("Password for student@10.2.100.14");
  expect(button(container, "Connect")).toBeTruthy();
});

test("the connection state is re-read while the dialog is open", async () => {
  vi.useFakeTimers();
  await render();
  const before = calls.filter((c) => c.path === "/remote-servers").length;
  servers = [{ ...SERVER, session: { connected: false } }];
  await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
  expect(calls.filter((c) => c.path === "/remote-servers").length).toBeGreaterThan(before);
  expect(container.textContent).toContain("Password for student@10.2.100.14");
});

test("with several servers it offers a list and defaults to the last one used", async () => {
  const other = { ...SERVER, name: "gpu-box", host: "10.0.0.5", session: { connected: false } };
  servers = [other, SERVER];
  window.localStorage.setItem("lelab.remoteServer", "cluster");
  await render();
  const select = container.querySelector("select")!;
  expect(Array.from(select.options).map((o) => o.value)).toEqual(["", "gpu-box", "cluster"]);
  expect(select.value).toBe("cluster");
  expect(container.textContent).toContain("cluster — student@10.2.100.14 · connected");
});

test("after sending with the Slurm files it lists the sbatch commands", async () => {
  vi.useFakeTimers();
  uploadResponse = () => {
    statuses = [{
      state: "done", dataset_repo_id: REPO, bytes_total: 1, bytes_done: 1, files_total: 1, files_done: 1,
      remote_path: "/home/student/projects/data/x", train_command: "lerobot-train ...",
      slurm_commands: ["DATASET=/home/student/projects/data/x sbatch /home/student/slurm/train_act.sbatch"],
      training_kit: { remote_dir: "/home/student", dataset_dir: "/home/student/projects/data", files: [], created_env: true, env_path: "/home/student/.env", serve_command: "sbatch /home/student/slurm/serve_policy.sbatch" },
    }];
    return json({ state: "running", dataset_repo_id: REPO, bytes_total: 1, bytes_done: 0, files_total: 1, files_done: 0 });
  };
  await render();
  await click(button(container, "Send"));
  await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
  expect(container.textContent).toContain("sbatch /home/student/slurm/train_act.sbatch");
  expect(container.textContent).toContain("Created .env from .env.example");
  expect(container.textContent).not.toContain("lerobot-train ...");
});

test("unticking the Slurm files leaves them out, and the choice is remembered", async () => {
  await render();
  await click(container.querySelector('[aria-label="Also send the Slurm training files"]') as HTMLElement);
  await click(button(container, "Send"));
  expect(calls.find((c) => c.path === "/remote-upload")!.body!.send_training_kit).toBe(false);
  expect(window.localStorage.getItem("lelab.sendTrainingKit")).toBe("false");
});
