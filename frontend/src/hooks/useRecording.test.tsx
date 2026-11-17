// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { useRecording } from "./useRecording";
import { RobotRecord } from "@/hooks/useRobots";

const mocks = vi.hoisted(() => ({ navigate: vi.fn(), remote: null as unknown }));
vi.mock("react-router-dom", () => ({ useNavigate: () => mocks.navigate }));
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ toast: vi.fn() }) }));
vi.mock("@/contexts/HfAuthContext", () => ({ useHfAuth: () => ({ auth: { status: "unauthenticated" } }) }));
vi.mock("@/hooks/useAvailableCameras", () => ({ useAvailableCameras: () => ({ cameras: [] }) }));
vi.mock("@/contexts/RemoteServerContext", () => ({ useOptionalRemoteServers: () => mocks.remote }));

const robot: RobotRecord = {
  name: "so101", leader_port: "leader", follower_port: "follower", leader_config: "l.json", follower_config: "f.json",
  is_clean: true, cameras: [],
};
let root: Root;
let recording: ReturnType<typeof useRecording>;
const Harness = () => {
  recording = useRecording(robot);
  return null;
};

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  window.localStorage.clear();
  mocks.navigate.mockReset();
  root = createRoot(document.createElement("div"));
});
afterEach(() => act(() => root.unmount()));

async function startRecording(send: boolean) {
  await act(async () => root.render(<Harness />));
  act(() => {
    recording.openForNew("pick_cube");
    recording.modalProps.setSingleTask("pick the cube");
    recording.modalProps.setSendToServer(send);
  });
  await act(async () => {
    await recording.modalProps.onStart();
  });
  return mocks.navigate.mock.calls[0];
}

test("asks for the upload after recording when a server is saved and the box is ticked", async () => {
  mocks.remote = { selected: { name: "cluster" } };
  const [path, options] = await startRecording(true);
  expect(path).toBe("/recording");
  expect(options.state.sendToServer).toBe(true);
  expect(recording.modalProps.sendServerName).toBe("cluster");
  expect(window.localStorage.getItem("lelab.sendToServerAfterRecording")).toBe("1");
});

test("never asks without a saved server", async () => {
  mocks.remote = null;
  const [, options] = await startRecording(true);
  expect(options.state.sendToServer).toBe(false);
  expect(recording.modalProps.sendServerName).toBeNull();
});
