import { useCallback, useEffect, useState } from "react";
import { CAMERA_ROTATIONS, type CameraRotation } from "@/lib/cameraConfig";
import type { BlendingRequestFields, InteractiveRequestFields } from "@/lib/inferenceApi";

export const COMMAND_TIME_MIN_S = 5;
export const COMMAND_TIME_MAX_S = 600;
export const BLEND_STEPS_MIN = 1;
export const BLEND_STEPS_MAX = 200;

/** Where each start dialog keeps the settings of its last run. */
export const LOCAL_RUN_SETTINGS_KEY = "lelab.runSettings.local";
export const REMOTE_RUN_SETTINGS_KEY = "lelab.runSettings.remote";

/** A camera bound to a policy camera: the browser's device id, and the cv2 index when that id is unknown. */
export interface RememberedCamera {
  device_id: string;
  camera_index: number;
}

export interface VideoChoice {
  main: string;
  corner: string | null;
}

/** Everything a start dialog is set to when a run starts. */
export interface RunSettings {
  task: string;
  durationS: number;
  interactive: boolean;
  commandTimeS: number;
  returnBetweenCommands: boolean;
  autoComplete: boolean;
  chunkBlending: boolean;
  blendSteps: number;
  /** Run on a server, under Advanced. null: left empty (no cap / the server's own value). */
  maxStep: number | null;
  nActionSteps: number | null;
  prefetch: number;
  /** Policy camera name → the physical camera bound to it. */
  cameras: Record<string, RememberedCamera>;
  /** Rotations picked in the dialog, by physical camera (see physicalCameraKey). */
  rotations: Record<string, CameraRotation>;
  /** The cameras picked for the video, when the layout wasn't obvious. */
  videoChoice: VideoChoice | null;
}

export const DEFAULT_RUN_SETTINGS: RunSettings = {
  task: "",
  durationS: 60,
  interactive: false,
  commandTimeS: 60,
  returnBetweenCommands: true,
  autoComplete: true,
  chunkBlending: false,
  blendSteps: 10,
  maxStep: 20,
  nActionSteps: null,
  prefetch: 10,
  cameras: {},
  rotations: {},
  videoChoice: null,
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);
const positiveInt = (value: unknown): value is number =>
  typeof value === "number" && Number.isInteger(value) && value >= 1;

/**
 * Settings read back from storage. Each field is checked on its own: a missing or malformed one (an older
 * LeLab, a hand edit) falls back to its default without losing the rest.
 */
export function parseRunSettings(raw: unknown): RunSettings {
  const d = DEFAULT_RUN_SETTINGS;
  if (!isRecord(raw)) return { ...d };
  const bool = (key: keyof RunSettings, fallback: boolean) =>
    typeof raw[key] === "boolean" ? (raw[key] as boolean) : fallback;
  const int = (key: keyof RunSettings, fallback: number) => (positiveInt(raw[key]) ? (raw[key] as number) : fallback);
  const optionalInt = (key: keyof RunSettings, fallback: number | null) =>
    raw[key] === null ? null : positiveInt(raw[key]) ? (raw[key] as number) : fallback;

  const cameras: Record<string, RememberedCamera> = {};
  if (isRecord(raw.cameras)) {
    for (const [name, camera] of Object.entries(raw.cameras)) {
      if (
        isRecord(camera) &&
        typeof camera.device_id === "string" &&
        typeof camera.camera_index === "number" &&
        Number.isInteger(camera.camera_index)
      ) {
        cameras[name] = { device_id: camera.device_id, camera_index: camera.camera_index };
      }
    }
  }
  const rotations: Record<string, CameraRotation> = {};
  if (isRecord(raw.rotations)) {
    for (const [key, rotation] of Object.entries(raw.rotations)) {
      if ((CAMERA_ROTATIONS as readonly unknown[]).includes(rotation)) rotations[key] = rotation as CameraRotation;
    }
  }
  const choice = raw.videoChoice;
  const videoChoice =
    isRecord(choice) && typeof choice.main === "string" && (choice.corner === null || typeof choice.corner === "string")
      ? { main: choice.main, corner: choice.corner as string | null }
      : null;

  return {
    task: typeof raw.task === "string" ? raw.task : d.task,
    durationS: int("durationS", d.durationS),
    interactive: bool("interactive", d.interactive),
    commandTimeS: int("commandTimeS", d.commandTimeS),
    returnBetweenCommands: bool("returnBetweenCommands", d.returnBetweenCommands),
    autoComplete: bool("autoComplete", d.autoComplete),
    chunkBlending: bool("chunkBlending", d.chunkBlending),
    blendSteps: int("blendSteps", d.blendSteps),
    maxStep: optionalInt("maxStep", d.maxStep),
    nActionSteps: optionalInt("nActionSteps", d.nActionSteps),
    prefetch: typeof raw.prefetch === "number" && Number.isInteger(raw.prefetch) && raw.prefetch >= 0 ? raw.prefetch : d.prefetch,
    cameras,
    rotations,
    videoChoice,
  };
}

export function loadRunSettings(key: string): RunSettings {
  try {
    return parseRunSettings(JSON.parse(window.localStorage.getItem(key) ?? "null"));
  } catch {
    return { ...DEFAULT_RUN_SETTINGS };
  }
}

export function saveRunSettings(key: string, settings: RunSettings): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(settings));
  } catch {
    // storage unavailable: the next run starts from the defaults
  }
}

/**
 * A start dialog's settings: filled in from its last run each time the dialog opens, and saved by `save()`
 * when a run starts (so settings changed and then cancelled don't stick).
 */
export function useRunSettings(storageKey: string, open: boolean) {
  const [settings, setSettings] = useState<RunSettings>(() => loadRunSettings(storageKey));
  useEffect(() => {
    if (open) setSettings(loadRunSettings(storageKey));
  }, [open, storageKey]);
  const update = useCallback((patch: Partial<RunSettings>) => setSettings((prev) => ({ ...prev, ...patch })), []);
  const save = (extra: Partial<RunSettings> = {}) => saveRunSettings(storageKey, { ...settings, ...extra });
  return { settings, update, save };
}

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

export function interactiveRequest(settings: RunSettings): InteractiveRequestFields {
  return settings.interactive
    ? {
        interactive: true,
        command_time_s: clamp(settings.commandTimeS, COMMAND_TIME_MIN_S, COMMAND_TIME_MAX_S),
        return_between_commands: settings.returnBetweenCommands,
        auto_complete: settings.autoComplete,
      }
    : {};
}

export function blendingRequest(settings: RunSettings): BlendingRequestFields {
  return {
    chunk_blending: settings.chunkBlending,
    blend_steps: clamp(settings.blendSteps, BLEND_STEPS_MIN, BLEND_STEPS_MAX),
  };
}
