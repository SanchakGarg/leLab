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

const dataset = (targetEpisodes: number | null) => ({
  repoId: "local/ds", numEpisodes: 12, targetEpisodes, fps: 30, imageFeatures: {},
});

async function appendTo(target: ReturnType<typeof dataset>) {
  mocks.remote = null;
  await act(async () => root.render(<Harness />));
  act(() => recording.openToAppend(target, "pick the cube"));
}

async function start() {
  await act(async () => {
    await recording.modalProps.onStart();
  });
  return mocks.navigate.mock.calls.at(-1)?.[1]?.state.recordingConfig;
}

test("resuming records the episodes the plan still needs", async () => {
  await appendTo(dataset(20));
  expect(recording.modalProps.totalEpisodes).toBe(20);
  const config = await start();
  expect(config.num_episodes).toBe(8);
  expect(config.resume).toBe(true);
  expect(config.dataset_repo_id).toBe("local/ds");
});

test("recording more asks for a higher total", async () => {
  await appendTo(dataset(null));
  expect(recording.modalProps.totalEpisodes).toBe(17);

  act(() => recording.modalProps.setTotalEpisodes(12));
  expect(await start()).toBeUndefined();

  act(() => recording.modalProps.setTotalEpisodes(14));
  expect((await start()).num_episodes).toBe(2);
});

test("a finished plan counts as recording more", async () => {
  await appendTo({ ...dataset(12) });
  expect(recording.modalProps.totalEpisodes).toBe(17);
});
