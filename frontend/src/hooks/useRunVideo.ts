import { useState } from "react";
import type { VideoChoice } from "@/hooks/useRunSettings";
import type { RunVideoRequestFields } from "@/lib/inferenceApi";
import { VideoLayout, defaultVideoLayout } from "@/lib/runVideo";

const VIDEO_KEY = "lelab.recordRunVideo";

const readRecordVideo = () => {
  try {
    return window.localStorage.getItem(VIDEO_KEY) === "1";
  } catch {
    return false;
  }
};

export interface RunVideo {
  cameras: string[];
  recordVideo: boolean;
  setRecordVideo: (value: boolean) => void;
  defaults: VideoLayout;
  main: string | null;
  corner: string | null;
  /** The user's own pick of cameras; null goes back to the remembered pick, or the default layout. */
  choose: (choice: VideoChoice | null) => void;
  /** The pick in effect when it isn't the default layout, to remember for the next run. */
  choice: VideoChoice | null;
  cameraLabel: (camera: string) => string;
  /** What the start request sends. */
  request: RunVideoRequestFields;
}

/**
 * Whether a model run records a video, and which of the policy's `cameras` it shows (one full frame,
 * another in the corner). `boundName` names the robot camera bound to a policy camera. The checkbox is
 * remembered across runs; `remembered` is the pick of the last run, used when this model has those cameras.
 */
export function useRunVideo(
  cameras: string[],
  boundName: (camera: string) => string | null | undefined,
  remembered: VideoChoice | null = null,
): RunVideo {
  const [recordVideo, setRecord] = useState(readRecordVideo);
  const [picked, choose] = useState<VideoChoice | null>(null);
  const fits = (pick: VideoChoice | null) =>
    !!pick &&
    cameras.includes(pick.main) &&
    (pick.corner === null || (pick.corner !== pick.main && cameras.includes(pick.corner)));
  const choice = fits(picked) ? picked : fits(remembered) ? remembered : null;
  const defaults = defaultVideoLayout(cameras, boundName);
  const main = choice?.main ?? defaults.main;
  const corner = choice ? choice.corner : defaults.corner;
  const setRecordVideo = (value: boolean) => {
    setRecord(value);
    try {
      window.localStorage.setItem(VIDEO_KEY, value ? "1" : "0");
    } catch {
      // storage unavailable
    }
  };
  const cameraLabel = (camera: string) => {
    const bound = boundName(camera);
    return bound && bound !== camera ? `${camera} (${bound})` : camera;
  };
  const on = recordVideo && cameras.length > 0;
  return {
    cameras,
    recordVideo,
    setRecordVideo,
    defaults,
    main,
    corner,
    choose,
    choice,
    cameraLabel,
    request: {
      record_video: on && !!main,
      video_camera: on ? main : null,
      video_corner_camera: on ? corner : null,
    },
  };
}
