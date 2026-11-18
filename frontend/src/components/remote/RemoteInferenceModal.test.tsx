// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import RemoteInferenceModal from "./RemoteInferenceModal";
import { RemoteServerProvider } from "@/contexts/RemoteServerContext";
import { RobotRecord } from "@/hooks/useRobots";
import { button, click, json, type } from "./testUtils";

const mocks = vi.hoisted(() => ({ fetcher: vi.fn(), navigate: vi.fn(), toast: vi.fn() }));
const available = [
  { index: 1, name: "front", deviceId: "usb-front", available: true },
  { index: 2, name: "wrist", deviceId: "usb-wrist", available: true },
];
vi.mock("@/contexts/ApiContext", () => ({ useApi: () => ({ baseUrl: "http://test", fetchWithHeaders: mocks.fetcher }) }));
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ toast: mocks.toast }) }));
vi.mock("react-router-dom", () => ({ useNavigate: () => mocks.navigate }));
vi.mock("@/hooks/useAvailableCameras", () => ({ useAvailableCameras: () => ({ cameras: available }) }));
vi.mock("@/hooks/useCameraStream", () => ({ useCameraStream: () => ({ videoRef: React.createRef<HTMLVideoElement>(), hasError: false }) }));
vi.mock("@/components/ui/select", async () => (await import("./testUtils")).uiMocks.select());
vi.mock("@/components/ui/dialog", async () => (await import("./testUtils")).uiMocks.dialog());

const SERVER = {
  name: "cluster", host: "10.2.100.14", port: 22, username: "student", auth: "password", key_path: "",
  remote_dataset_dir: "", remote_work_dir: "", session: { connected: true },
};
const HELLO = {
  type: "hello", name: "act (outputs/act_pick/checkpoints/last/pretrained_model)", cameras: ["front", "wrist"],
  image_shapes: { front: [480, 640, 3], wrist: [240, 320, 3] }, chunk_size: 50, n_action_steps: 50, fps: 30,
  state_dim: 6, action_dim: 6, address: "10.2.0.7:5555 via cluster",
};
const robot: RobotRecord = {
  name: "so101", leader_port: "leader", follower_port: "/dev/follower", leader_config: "l.json", follower_config: "f.json",
  is_clean: true,
  cameras: [{ id: "front", name: "front", type: "opencv", device_id: "usb-front", camera_index: 1, width: 640, height: 480, rotation: 0 }],
};
let calls: { path: string; body: Record<string, unknown> | null }[];
let root: Root;
let container: HTMLDivElement;

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  window.localStorage.clear();
  container = document.createElement("div");
  root = createRoot(container);
  calls = [];
  mocks.fetcher.mockReset(); mocks.navigate.mockReset(); mocks.toast.mockReset();
  mocks.fetcher.mockImplementation(async (url: string, init: RequestInit = {}) => {
    const path = url.replace("http://test", "");
    calls.push({ path, body: init.body ? JSON.parse(String(init.body)) : null });
    if (path === "/remote-servers") return json({ servers: [SERVER] });
    if (path === "/remote-policy/probe") return json(HELLO);
    if (path === "/start-remote-inference") return json({ message: "started", policy_ref: "x" });
    return json({}, 404);
  });
});
afterEach(() => {
  act(() => root.unmount());
  vi.useRealTimers();
});

const cameraSelects = () =>
  Array.from(container.querySelectorAll("select")).filter((s) => Array.from(s.options).some((o) => o.text.includes("#")));

test("connects to the model, sizes the cameras from it, and starts the arm with them", async () => {
  await act(async () =>
    root.render(
      <RemoteServerProvider>
        <RemoteInferenceModal open onOpenChange={vi.fn()} robot={robot} />
      </RemoteServerProvider>,
    ),
  );
  type(container.querySelector<HTMLInputElement>("#policy-host")!, "10.2.0.7");
  await click(button(container, "Connect to the model"));

  expect(calls.find((c) => c.path === "/remote-policy/probe")!.body).toEqual({
    server: "cluster", connection: "tunnel", policy_host: "10.2.0.7", policy_port: 5555,
  });
  expect(container.textContent).toContain("640×480");
  expect(container.textContent).toContain("320×240");
  // front binds itself (same name as a robot camera); wrist is still open, so Start waits.
  expect(cameraSelects().map((s) => s.value)).toEqual(["1", ""]);
  expect(button(container, "Start").disabled).toBe(true);

  act(() => {
    const wrist = cameraSelects()[1];
    wrist.value = "2";
    wrist.dispatchEvent(new Event("change", { bubbles: true }));
  });
  vi.useFakeTimers();
  await act(async () => {
    button(container, "Start").click();
    await vi.advanceTimersByTimeAsync(300);
  });

  const start = calls.find((c) => c.path === "/start-remote-inference")!.body!;
  expect(start).toMatchObject({
    server: "cluster", connection: "tunnel", policy_host: "10.2.0.7", policy_port: 5555,
    follower_port: "/dev/follower", follower_config: "f.json", fps: 30, max_relative_target: 20, prefetch: 10,
  });
  expect(start.cameras).toMatchObject({
    front: { camera_index: 1, width: 640, height: 480 },
    wrist: { camera_index: 2, width: 320, height: 240 },
  });
  expect(mocks.navigate).toHaveBeenCalledWith("/inference");
});

const renderAndConnect = async () => {
  await act(async () =>
    root.render(
      <RemoteServerProvider>
        <RemoteInferenceModal open onOpenChange={vi.fn()} robot={robot} />
      </RemoteServerProvider>,
    ),
  );
  type(container.querySelector<HTMLInputElement>("#policy-host")!, "10.2.0.7");
  await click(button(container, "Connect to the model"));
};

test("records a video with the wrist camera in the corner", async () => {
  await renderAndConnect();
  act(() => {
    const wrist = cameraSelects()[1];
    wrist.value = "2";
    wrist.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await click(container.querySelector<HTMLButtonElement>("#remote-record-video")!);
  expect(container.textContent).toContain("front full frame, wrist in the corner");
  // Two cameras, one of them the wrist camera: nothing to choose.
  expect(container.textContent).not.toContain("In the corner");
  expect(window.localStorage.getItem("lelab.recordRunVideo")).toBe("1");

  vi.useFakeTimers();
  await act(async () => {
    button(container, "Start").click();
    await vi.advanceTimersByTimeAsync(300);
  });
  expect(calls.find((c) => c.path === "/start-remote-inference")!.body).toMatchObject({
    record_video: true, video_camera: "front", video_corner_camera: "wrist",
  });
});

test("asks which cameras go in the video when there are more than two", async () => {
  window.localStorage.setItem("lelab.recordRunVideo", "1");
  const three = { ...HELLO, cameras: ["top", "side", "wrist"], image_shapes: { top: [480, 640, 3], side: [480, 640, 3], wrist: [240, 320, 3] } };
  mocks.fetcher.mockImplementation(async (url: string, init: RequestInit = {}) => {
    const path = url.replace("http://test", "");
    calls.push({ path, body: init.body ? JSON.parse(String(init.body)) : null });
    if (path === "/remote-servers") return json({ servers: [SERVER] });
    if (path === "/remote-policy/probe") return json(three);
    return json({}, 404);
  });
  await renderAndConnect();

  expect(container.textContent).toContain("Full view");
  expect(container.textContent).toContain("In the corner");
  const videoSelects = () =>
    Array.from(container.querySelectorAll("select")).filter((s) =>
      Array.from(s.options).some((o) => ["top", "__none__"].includes(o.value)),
    );
  // Defaults: the first other camera full frame, the wrist camera in the corner.
  expect(videoSelects().map((s) => s.value)).toEqual(["top", "wrist"]);

  act(() => {
    const corner = videoSelects()[1];
    corner.value = "__none__";
    corner.dispatchEvent(new Event("change", { bubbles: true }));
  });
  expect(videoSelects().map((s) => s.value)).toEqual(["top", "__none__"]);
  // The full-frame camera is never offered for the corner too.
  expect(Array.from(videoSelects()[1].options).map((o) => o.value)).not.toContain("top");
});

test("starts an interactive session: no task or duration, a time per command instead", async () => {
  await renderAndConnect();
  act(() => {
    const wrist = cameraSelects()[1];
    wrist.value = "2";
    wrist.dispatchEvent(new Event("change", { bubbles: true }));
  });
  expect(container.querySelector("#remote-task")).not.toBeNull();
  await click(container.querySelector<HTMLButtonElement>("#remote-interactive")!);
  expect(container.querySelector("#remote-task")).toBeNull();
  expect(container.querySelector("#remote-duration")).toBeNull();
  type(container.querySelector<HTMLInputElement>("#remote-command-time")!, "45");
  await click(container.querySelector<HTMLButtonElement>("#remote-return-between")!);

  vi.useFakeTimers();
  await act(async () => {
    button(container, "Start session").click();
    await vi.advanceTimersByTimeAsync(300);
  });
  expect(calls.find((c) => c.path === "/start-remote-inference")!.body).toMatchObject({
    interactive: true, command_time_s: 45, return_between_commands: false, auto_complete: true, task: "",
  });
  expect(mocks.navigate).toHaveBeenCalledWith("/inference");
  expect(JSON.parse(window.localStorage.getItem("lelab.runSettings.remote")!)).toMatchObject({
    interactive: true, commandTimeS: 45, returnBetweenCommands: false,
  });
});

const bindWrist = () =>
  act(() => {
    const wrist = cameraSelects()[1];
    wrist.value = "2";
    wrist.dispatchEvent(new Event("change", { bubbles: true }));
  });

const openAdvanced = () => click(button(container, "Advanced"));

const startRun = async () => {
  vi.useFakeTimers();
  await act(async () => {
    Array.from(container.querySelectorAll("button")).find((b) => /^Start/.test(b.textContent ?? ""))!.click();
    await vi.advanceTimersByTimeAsync(300);
  });
  vi.useRealTimers();
  return calls.filter((c) => c.path === "/start-remote-inference").at(-1)!.body!;
};

test("blends chunks over the chosen number of actions, and warns when prefetch leaves nothing to blend", async () => {
  await renderAndConnect();
  bindWrist();
  expect(container.querySelector("#remote-chunk-blending-steps")).toBeNull();
  await click(container.querySelector<HTMLButtonElement>("#remote-chunk-blending")!);
  type(container.querySelector<HTMLInputElement>("#remote-chunk-blending-steps")!, "6");
  expect(container.textContent).toContain("About 0.20 s at 30 Hz");
  await openAdvanced();
  type(container.querySelector<HTMLInputElement>("#remote-prefetch")!, "0");
  expect(container.textContent).toContain("there is nothing to blend");
  type(container.querySelector<HTMLInputElement>("#remote-prefetch")!, "4");
  expect(container.textContent).toContain("At most 4 actions are left");

  expect(await startRun()).toMatchObject({ chunk_blending: true, blend_steps: 6, prefetch: 4 });
});

test("the next run starts from the settings of the last one, cameras included", async () => {
  await renderAndConnect();
  bindWrist();
  type(container.querySelector<HTMLInputElement>("#remote-task")!, "stack the cubes");
  type(container.querySelector<HTMLInputElement>("#remote-duration")!, "90");
  await openAdvanced();
  type(container.querySelector<HTMLInputElement>("#remote-maxstep")!, "");
  await click(container.querySelector<HTMLButtonElement>("#remote-chunk-blending")!);
  const first = await startRun();
  expect(first).toMatchObject({ task: "stack the cubes", duration_s: 90, max_relative_target: null, chunk_blending: true });

  // A new dialog (the page was left and opened again): everything comes back, and still can be changed.
  act(() => root.unmount());
  root = createRoot(container);
  await renderAndConnect();
  expect(container.querySelector<HTMLInputElement>("#remote-task")!.value).toBe("stack the cubes");
  expect(container.querySelector<HTMLInputElement>("#remote-duration")!.value).toBe("90");
  await openAdvanced();
  expect(container.querySelector<HTMLInputElement>("#remote-maxstep")!.value).toBe("");
  expect(container.querySelector<HTMLButtonElement>("#remote-chunk-blending")!.getAttribute("data-state")).toBe("checked");
  expect(cameraSelects().map((s) => s.value)).toEqual(["1", "2"]); // wrist bound again without asking
  type(container.querySelector<HTMLInputElement>("#remote-task")!, "unstack them");
  expect(await startRun()).toMatchObject({
    task: "unstack them", duration_s: 90, chunk_blending: true,
    cameras: { front: { camera_index: 1 }, wrist: { camera_index: 2 } },
  });
});

test("cancelled changes are not remembered", async () => {
  await renderAndConnect();
  type(container.querySelector<HTMLInputElement>("#remote-task")!, "never started");
  act(() => root.unmount());
  root = createRoot(container);
  await renderAndConnect();
  expect(container.querySelector<HTMLInputElement>("#remote-task")!.value).toBe("");
  expect(window.localStorage.getItem("lelab.runSettings.remote")).toBeNull();
});
