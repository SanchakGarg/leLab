import { ApiError, Fetcher, apiRequest } from "./apiClient";
import { CameraRequest } from "./cameraConfig";
import { ImageFeatures } from "./checkpointsApi";

/** How LeLab logs into a server. Passwords are sent to connect and never saved. */
export type RemoteAuth = "password" | "key" | "agent";

export interface RemoteServerProfile {
  host: string;
  port: number;
  username: string;
  auth: RemoteAuth;
  key_path: string;
  /** Folder on the server that receives datasets (relative = under the home). */
  remote_dataset_dir: string;
  /** Folder on the server for the policy server script. */
  remote_work_dir: string;
}

export interface RemoteSession {
  connected: boolean;
  connected_at?: number;
  home?: string | null;
  in_use?: boolean;
}

export interface RemoteServer extends RemoteServerProfile {
  name: string;
  session: RemoteSession;
}

export const emptyProfile = (): RemoteServerProfile => ({
  host: "",
  port: 22,
  username: "",
  auth: "password",
  key_path: "",
  remote_dataset_dir: "",
  remote_work_dir: "",
});

const enc = encodeURIComponent;

export async function listRemoteServers(baseUrl: string, fetcher: Fetcher): Promise<RemoteServer[]> {
  const body = await apiRequest<{ servers: RemoteServer[] }>(baseUrl, fetcher, "/remote-servers", {
    action: "List servers",
  });
  return body.servers;
}

export async function saveRemoteServer(
  baseUrl: string,
  fetcher: Fetcher,
  name: string,
  profile: RemoteServerProfile,
): Promise<RemoteServerProfile> {
  return apiRequest(baseUrl, fetcher, `/remote-servers/${enc(name)}`, {
    method: "PUT",
    body: profile,
    action: "Save server",
  });
}

export async function deleteRemoteServer(baseUrl: string, fetcher: Fetcher, name: string): Promise<void> {
  await apiRequest(baseUrl, fetcher, `/remote-servers/${enc(name)}`, {
    method: "DELETE",
    action: "Delete server",
  });
}

export async function connectRemoteServer(
  baseUrl: string,
  fetcher: Fetcher,
  name: string,
  body: { password?: string; trust_fingerprint?: string },
): Promise<RemoteSession> {
  return apiRequest<RemoteSession>(baseUrl, fetcher, `/remote-servers/${enc(name)}/connect`, {
    method: "POST",
    body,
    action: "Connect",
  });
}

export async function disconnectRemoteServer(baseUrl: string, fetcher: Fetcher, name: string): Promise<void> {
  await apiRequest(baseUrl, fetcher, `/remote-servers/${enc(name)}/disconnect`, {
    method: "POST",
    action: "Disconnect",
  });
}

export async function installPolicyServer(
  baseUrl: string,
  fetcher: Fetcher,
  name: string,
  remoteDir: string,
): Promise<{ remote_path: string; command: string }> {
  return apiRequest(baseUrl, fetcher, `/remote-servers/${enc(name)}/install-policy-server`, {
    method: "POST",
    body: { remote_dir: remoteDir },
    action: "Copy the policy server",
  });
}

/** The server's host key, when a connect needs the user to confirm it first. */
export interface HostKeyChallenge {
  fingerprint: string;
  keyType: string;
  message: string;
}

export function hostKeyChallenge(error: unknown): HostKeyChallenge | null {
  if (!(error instanceof ApiError) || error.data?.code !== "unknown_host_key") return null;
  return {
    fingerprint: String(error.data.fingerprint ?? ""),
    keyType: String(error.data.key_type ?? ""),
    message: error.detail ?? "",
  };
}

export const errorText = (error: unknown): string =>
  error instanceof ApiError ? error.detail ?? error.message : error instanceof Error ? error.message : String(error);

// ------------------------------------------------------------------------------------------- upload
export type RemoteUploadState = "idle" | "running" | "done" | "error" | "cancelled";

export interface RemoteUploadStatus {
  state: RemoteUploadState;
  server?: string;
  dataset_repo_id?: string;
  remote_name?: string;
  remote_path?: string | null;
  bytes_total?: number;
  bytes_done?: number;
  files_total?: number;
  files_done?: number;
  files_skipped?: number;
  current_file?: string | null;
  error?: string | null;
  train_command?: string | null;
}

export interface RemoteUploadRequest {
  server: string;
  dataset_repo_id: string;
  remote_dir: string;
  remote_name: string;
}

export async function startRemoteUpload(
  baseUrl: string,
  fetcher: Fetcher,
  request: RemoteUploadRequest,
): Promise<RemoteUploadStatus> {
  return apiRequest(baseUrl, fetcher, "/remote-upload", {
    method: "POST",
    body: request,
    action: "Send to server",
  });
}

export async function getRemoteUploadStatus(baseUrl: string, fetcher: Fetcher): Promise<RemoteUploadStatus> {
  return apiRequest(baseUrl, fetcher, "/remote-upload/status", { action: "Upload status" });
}

export async function cancelRemoteUpload(baseUrl: string, fetcher: Fetcher): Promise<void> {
  await apiRequest(baseUrl, fetcher, "/remote-upload/cancel", { method: "POST", action: "Cancel upload" });
}

// ----------------------------------------------------------------------------------------- inference
export type PolicyConnectionKind = "tunnel" | "direct";

export interface PolicyConnection {
  server: string | null;
  connection: PolicyConnectionKind;
  /** Where the policy server listens: the GPU node's IP as seen from the server (tunnel), or from here (direct). */
  policy_host: string;
  policy_port: number;
}

export interface PolicyHello {
  name: string;
  cameras: string[];
  /** camera → [height, width, channels]; absent on older policy servers. */
  image_shapes?: Record<string, number[]>;
  state_dim?: number;
  action_dim?: number;
  chunk_size: number;
  n_action_steps: number;
  fps: number;
  address: string;
}

export async function probePolicyServer(
  baseUrl: string,
  fetcher: Fetcher,
  connection: PolicyConnection,
): Promise<PolicyHello> {
  return apiRequest(baseUrl, fetcher, "/remote-policy/probe", {
    method: "POST",
    body: connection,
    action: "Connect to the policy server",
  });
}

export interface StartRemoteInferenceRequest extends PolicyConnection {
  follower_port: string;
  follower_config: string;
  task: string;
  cameras: Record<string, CameraRequest>;
  duration_s: number;
  fps?: number | null;
  n_action_steps?: number | null;
  prefetch?: number;
  max_relative_target?: number | null;
}

export async function startRemoteInference(
  baseUrl: string,
  fetcher: Fetcher,
  request: StartRemoteInferenceRequest,
): Promise<{ message: string; policy_ref: string }> {
  return apiRequest(baseUrl, fetcher, "/start-remote-inference", {
    method: "POST",
    body: request,
    action: "Start remote inference",
  });
}

/** The resolution each camera must have: from the server, else a same-named robot camera, else 640×480. */
export function imageFeaturesFromHello(
  hello: PolicyHello,
  robotCameras: { name: string; width?: number | null; height?: number | null }[] = [],
): ImageFeatures {
  const features: ImageFeatures = {};
  for (const name of hello.cameras) {
    const shape = hello.image_shapes?.[name];
    if (shape && shape.length >= 2) {
      features[name] = { height: shape[0], width: shape[1] };
      continue;
    }
    const cam = robotCameras.find((c) => c.name.toLowerCase() === name.toLowerCase());
    features[name] = { height: cam?.height || 480, width: cam?.width || 640 };
  }
  return features;
}

export const formatBytes = (n: number | undefined): string => {
  const v = n ?? 0;
  if (v < 1024) return `${v} B`;
  if (v < 1024 ** 2) return `${(v / 1024).toFixed(1)} KB`;
  if (v < 1024 ** 3) return `${(v / 1024 ** 2).toFixed(1)} MB`;
  return `${(v / 1024 ** 3).toFixed(2)} GB`;
};
