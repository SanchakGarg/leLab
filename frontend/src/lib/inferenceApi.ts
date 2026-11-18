import { Fetcher, apiRequest } from "./apiClient";
import { CameraRequest } from "./cameraConfig";
import type { ArmLinkIssue } from "@/components/control/ArmLinkNotice";

/** An interactive session: no task or duration up front; commands are sent one at a time while it runs. */
export interface InteractiveRequestFields {
  interactive?: boolean;
  /** Longest a command runs, from its first action. */
  command_time_s?: number;
  /** Back to the start pose after each command (the session's end always goes back). */
  return_between_commands?: boolean;
  /** End a command as done once the arm has moved and then stays still. */
  auto_complete?: boolean;
}

/**
 * Chunk blending: crossfade into each new action chunk over its first `blend_steps` actions, from the actions
 * of the previous chunk not played yet, so the switch doesn't jerk the arm. Local runs with it go through
 * LeLab's own control loop (lerobot-rollout doesn't blend).
 */
export interface BlendingRequestFields {
  chunk_blending?: boolean;
  blend_steps?: number;
}

/** Record a video: `video_camera` full frame, `video_corner_camera` inset in a corner (policy camera names). */
export interface RunVideoRequestFields {
  record_video?: boolean;
  video_camera?: string | null;
  video_corner_camera?: string | null;
}

export interface StartInferenceRequest
  extends InteractiveRequestFields,
    BlendingRequestFields,
    RunVideoRequestFields {
  follower_port: string;
  follower_config: string;
  policy_ref: string;
  task: string;
  cameras: Record<string, CameraRequest>;
  duration_s: number;
}

export type CommandStatus = "pending" | "running" | "completed" | "not_completed" | "stopped" | "failed";

export interface InteractiveCommand {
  id: number;
  text: string;
  status: CommandStatus;
  /** How it ended: "arm settled", "marked done", "time limit", "stopped", "session ended", or the error. */
  how: string | null;
  submitted_at: number;
  started_at: number | null;
  ended_at: number | null;
  elapsed_s: number;
}

/** "3 commands: 2 done, 1 didn't complete" */
export function summarizeCommands(commands: InteractiveCommand[]): string {
  if (commands.length === 0) return "No commands were run.";
  const labels: [CommandStatus, string][] = [
    ["completed", "done"],
    ["not_completed", "didn't complete"],
    ["stopped", "stopped"],
    ["failed", "failed"],
  ];
  const parts = labels
    .map(([status, label]) => [commands.filter((c) => c.status === status).length, label] as const)
    .filter(([n]) => n > 0)
    .map(([n, label]) => `${n} ${label}`);
  return `${commands.length} command${commands.length === 1 ? "" : "s"}: ${parts.join(", ")}.`;
}

export interface InteractiveStatus {
  state: "setting up" | "ready" | "running" | "returning";
  /** While setting up a local session: what it is doing. */
  phase?: string;
  command_time_s: number;
  return_between_commands: boolean;
  auto_complete: boolean;
  /** Switchable between commands (setChunkBlending). */
  chunk_blending: boolean;
  blend_steps: number;
  /** Every command of the session, oldest first. */
  commands: InteractiveCommand[];
}

export interface InferenceStatus {
  inference_active: boolean;
  started_at: number | null;
  rollout_started_at: number | null;
  elapsed_s: number;
  rollout_elapsed_s: number;
  duration_s: number | null;
  policy_ref: string | null;
  log_path: string | null;
  exited?: boolean;
  exit_code?: number | null;
  outcome?: "ok" | "ran_with_warning" | "failed";
  error?: string | null;
  hint?: string | null;
  /** "remote" when the policy runs on a remote server (lelab_policy_server.py). */
  mode?: "local" | "remote";
  remote?: RemoteRunStats;
  returned_to_start?: boolean | null;
  /** The follower's USB link failing now, or for a while recently (remote runs). */
  arm_link_issues?: ArmLinkIssue[];
  /** The run's video, when it records one (remote runs and interactive sessions). */
  video?: RunVideoInfo | null;
  /** Set for an interactive session. */
  interactive?: InteractiveStatus | null;
}

export interface RunVideoInfo {
  recording: boolean;
  /** Set once the file is saved: fetch it from /run-videos/{name}. */
  name: string | null;
  path: string | null;
  camera: string;
  corner_camera: string | null;
  frames: number;
  dropped: number;
  error: string | null;
}

export interface RemoteRunStats {
  address: string;
  policy: string | null;
  phase: string;
  steps: number;
  fps: number;
  chunks: number;
  rtt_ms_median: number | null;
  infer_ms_median: number | null;
  wait_s: number;
  actions_per_chunk: number;
  /** Follower reads/writes that failed and were skipped (USB glitches). */
  arm_skipped?: number;
  /** Chunk blending window in actions (0: off), and how many actions were blended so far. */
  blend_steps?: number;
  blended?: number;
}

export async function startInference(
  baseUrl: string,
  fetcher: Fetcher,
  request: StartInferenceRequest,
): Promise<{ message: string; log_path: string }> {
  return apiRequest<{ message: string; log_path: string }>(
    baseUrl,
    fetcher,
    "/start-inference",
    { method: "POST", body: request, action: "Start inference" },
  );
}

export async function stopInference(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<{ message: string }> {
  return apiRequest<{ message: string }>(baseUrl, fetcher, "/stop-inference", {
    method: "POST",
    action: "Stop inference",
  });
}

export async function getInferenceStatus(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<InferenceStatus> {
  return apiRequest<InferenceStatus>(baseUrl, fetcher, "/inference-status", {
    signal,
    action: "Get inference status",
  });
}

/** Run a command (the policy's task) in the interactive session. */
export async function sendInferenceCommand(
  baseUrl: string,
  fetcher: Fetcher,
  text: string,
): Promise<{ message: string; command: InteractiveCommand }> {
  return apiRequest(baseUrl, fetcher, "/inference-command", {
    method: "POST",
    body: { text },
    action: "Send the command",
  });
}

/** Done (completed) or Stop command for the command that is running. */
export async function endInferenceCommand(
  baseUrl: string,
  fetcher: Fetcher,
  completed: boolean,
): Promise<{ message: string }> {
  return apiRequest(baseUrl, fetcher, "/inference-command/end", {
    method: "POST",
    body: { completed },
    action: completed ? "Mark the command done" : "Stop the command",
  });
}

/** Switch chunk blending of the interactive session; only between commands. */
export async function setChunkBlending(
  baseUrl: string,
  fetcher: Fetcher,
  chunkBlending: boolean,
  blendSteps?: number,
): Promise<{ message: string; chunk_blending: boolean; blend_steps: number }> {
  return apiRequest(baseUrl, fetcher, "/inference-settings", {
    method: "POST",
    body: blendSteps === undefined ? { chunk_blending: chunkBlending } : { chunk_blending: chunkBlending, blend_steps: blendSteps },
    action: chunkBlending ? "Turn chunk blending on" : "Turn chunk blending off",
  });
}
