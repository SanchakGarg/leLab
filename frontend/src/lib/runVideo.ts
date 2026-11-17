// Which cameras a model run's video shows: one full frame, another (the wrist camera) in a corner.
// Mirrors default_layout() in lelab/run_video.py, but also looks at the robot camera each
// policy camera is bound to, since the policy's own names can be generic.

const WRIST_HINTS = ["wrist", "hand", "gripper"];

export const isWristCamera = (...names: (string | null | undefined)[]) =>
  names.some((name) => !!name && WRIST_HINTS.some((hint) => name.toLowerCase().includes(hint)));

export interface VideoLayout {
  main: string | null;
  corner: string | null;
  /** Nothing to choose: one camera, or two of which one is clearly the wrist camera. */
  obvious: boolean;
}

/** The default layout for these policy cameras; `boundName` names the robot camera bound to one. */
export function defaultVideoLayout(
  cameras: string[],
  boundName: (camera: string) => string | null | undefined = () => null,
): VideoLayout {
  if (cameras.length === 0) return { main: null, corner: null, obvious: true };
  if (cameras.length === 1) return { main: cameras[0], corner: null, obvious: true };
  const wrist = cameras.find((camera) => isWristCamera(camera, boundName(camera))) ?? null;
  if (!wrist) return { main: cameras[0], corner: null, obvious: false };
  const main = cameras.find((camera) => camera !== wrist) ?? null;
  return { main, corner: wrist, obvious: cameras.length === 2 };
}

/** Where the backend serves a saved run video. */
export const runVideoUrl = (baseUrl: string, name: string) =>
  `${baseUrl}/run-videos/${encodeURIComponent(name)}`;
