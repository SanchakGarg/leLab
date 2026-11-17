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
