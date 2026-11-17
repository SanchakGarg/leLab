// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import RecordingModal from "./RecordingModal";
import { RobotRecord } from "@/hooks/useRobots";
import { AppendTarget } from "@/hooks/useRecording";

vi.mock("@/contexts/HfAuthContext", () => ({ useHfAuth: () => ({ auth: { status: "unauthenticated" } }) }));
vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ open, children }: { open: boolean; children: React.ReactNode }) => (open ? <div>{children}</div> : null),
  DialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogTitle: ({ children }: { children: React.ReactNode }) => <h2>{children}</h2>,
  DialogDescription: ({ children }: { children: React.ReactNode }) => <p>{children}</p>,
}));

const robot: RobotRecord = {
  name: "so101", leader_port: "leader", follower_port: "follower", leader_config: "l.json", follower_config: "f.json",
  is_clean: true, cameras: [],
};
let root: Root;
let container: HTMLDivElement;
beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  container = document.createElement("div");
  root = createRoot(container);
});
afterEach(() => act(() => root.unmount()));

function render(appendTo: AppendTarget, totalEpisodes: number) {
  const noop = () => {};
  act(() =>
    root.render(
      <RecordingModal
        open
        onOpenChange={noop}
        robot={robot}
        appendTo={appendTo}
        cameraBindings={{ allBound: true } as never}
        datasetName=""
        setDatasetName={noop}
        singleTask="pick"
        setSingleTask={noop}
        numEpisodes={5}
        setNumEpisodes={noop}
        totalEpisodes={totalEpisodes}
        setTotalEpisodes={noop}
        episodeTimeS={60}
        setEpisodeTimeS={noop}
        resetTimeS={15}
        setResetTimeS={noop}
        streamingEncoding
        setStreamingEncoding={noop}
        cameras={[]}
        setCameras={noop}
        onStart={noop}
      />,
    ),
  );
}

const startButton = () =>
  Array.from(container.querySelectorAll("button")).find((b) => /(Start|Resume) Recording/.test(b.textContent ?? ""))!;
const target = (targetEpisodes: number | null): AppendTarget => ({
  repoId: "local/ds", numEpisodes: 12, targetEpisodes, fps: 30, imageFeatures: {},
});

test("an unfinished dataset offers to resume up to its plan", () => {
  render(target(20), 20);
  expect(container.querySelector("h2")?.textContent).toBe("Resume Recording");
  expect(container.textContent).toContain("12 of the 20 episodes planned");
  expect(container.textContent).toContain("8 more episodes will be recorded.");
  expect(startButton().textContent).toBe("Resume Recording");
  expect(startButton().disabled).toBe(false);
});

test("a finished dataset records more up to a higher total", () => {
  render(target(12), 13);
  expect(container.querySelector("h2")?.textContent).toBe("Record More Episodes");
  expect(container.textContent).toContain("1 more episode will be recorded.");
  expect(startButton().textContent).toBe("Start Recording");
});

test("a total that isn't higher than the dataset can't start", () => {
  render(target(null), 12);
  expect(container.textContent).toContain("Enter more than the 12 the dataset already has.");
  expect(startButton().disabled).toBe(true);
});
