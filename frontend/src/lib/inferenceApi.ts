import { Fetcher, apiRequest } from "./apiClient";
import { CameraRequest } from "./cameraConfig";

export interface StartInferenceRequest {
  follower_port: string;
  follower_config: string;
  policy_ref: string;
  task: string;
  cameras: Record<string, CameraRequest>;
  duration_s: number;
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
