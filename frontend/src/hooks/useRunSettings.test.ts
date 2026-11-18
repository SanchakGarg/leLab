// @vitest-environment jsdom
import { beforeEach, expect, test } from "vitest";
import {
  DEFAULT_RUN_SETTINGS,
  blendingRequest,
  interactiveRequest,
  loadRunSettings,
  parseRunSettings,
  saveRunSettings,
} from "./useRunSettings";

beforeEach(() => window.localStorage.clear());

test("nothing saved yet: the defaults", () => {
  expect(loadRunSettings("lelab.runSettings.local")).toEqual(DEFAULT_RUN_SETTINGS);
  expect(DEFAULT_RUN_SETTINGS).toMatchObject({ durationS: 60, commandTimeS: 60, chunkBlending: false, blendSteps: 10 });
});

test("saved settings come back as they were, empty fields included", () => {
  const saved = {
    ...DEFAULT_RUN_SETTINGS,
    task: "stack the cubes",
    interactive: true,
    chunkBlending: true,
    blendSteps: 6,
    maxStep: null,
    nActionSteps: 25,
    prefetch: 0,
    cameras: { wrist: { device_id: "usb-wrist", camera_index: 2 } },
    rotations: { "device:usb-wrist": 90 as const },
    videoChoice: { main: "top", corner: null },
  };
  saveRunSettings("k", saved);
  expect(loadRunSettings("k")).toEqual(saved);
});

test("a malformed field falls back to its default without losing the others", () => {
  const parsed = parseRunSettings({
    task: 42,
    durationS: -5,
    chunkBlending: true,
    blendSteps: 2.5,
    prefetch: "ten",
    maxStep: 0,
    cameras: { front: { device_id: "a", camera_index: 1 }, broken: { device_id: 3 } },
    rotations: { "device:a": 45, "device:b": 270 },
    videoChoice: { main: 7 },
  });
  expect(parsed).toMatchObject({
    task: "",
    durationS: 60,
    chunkBlending: true,
    blendSteps: 10,
    prefetch: 10,
    maxStep: 20,
    cameras: { front: { device_id: "a", camera_index: 1 } },
    rotations: { "device:b": 270 },
    videoChoice: null,
  });
  expect(parseRunSettings("not an object")).toEqual(DEFAULT_RUN_SETTINGS);
  window.localStorage.setItem("k", "{not json");
  expect(loadRunSettings("k")).toEqual(DEFAULT_RUN_SETTINGS);
});

test("the request fields: interactive ones only for a session, values kept in range", () => {
  expect(interactiveRequest(DEFAULT_RUN_SETTINGS)).toEqual({});
  expect(interactiveRequest({ ...DEFAULT_RUN_SETTINGS, interactive: true, commandTimeS: 2 })).toEqual({
    interactive: true,
    command_time_s: 5,
    return_between_commands: true,
    auto_complete: true,
  });
  expect(blendingRequest(DEFAULT_RUN_SETTINGS)).toEqual({ chunk_blending: false, blend_steps: 10 });
  expect(blendingRequest({ ...DEFAULT_RUN_SETTINGS, chunkBlending: true, blendSteps: 999 })).toEqual({
    chunk_blending: true,
    blend_steps: 200,
  });
});
