import { useEffect, useState } from "react";
import { RobotRecord } from "@/hooks/useRobots";
import { useAvailableCameras } from "@/hooks/useAvailableCameras";
import { ImageFeatures } from "@/lib/checkpointsApi";
import type { RememberedCamera } from "@/hooks/useRunSettings";
import {
  CameraRequest,
  CameraRotation,
  liveCameraFor,
  PhysicalCamera,
  physicalCameraKey,
  rotatedDimensions,
  rotationForCamera,
  serializeCamera,
} from "@/lib/cameraConfig";

export interface RememberedBindings {
  cameras: Record<string, RememberedCamera>;
  rotations: Record<string, CameraRotation>;
}

const NOTHING_REMEMBERED: RememberedBindings = { cameras: {}, rotations: {} };

/**
 * Binds each expected camera to a physical one. The camera it was bound to on the last run comes first
 * (`remembered`, when still plugged in), then a robot camera with the same name; the user picks the rest.
 * Pass the `remembered` object as stored (a new object each render would rebind every time).
 */
export function useCameraBindings(
  imageFeatures: ImageFeatures | null,
  robot: RobotRecord | null,
  enabled: boolean,
  remembered: RememberedBindings = NOTHING_REMEMBERED,
) {
  // Per expected camera name → user-selected physical camera index (or null).
  const [bindings, setBindings] = useState<Record<string, number | null>>({});
  // User rotation choice per physical camera, over the one saved with the robot.
  const [rotations, setRotations] = useState<Record<string, CameraRotation>>({});
  const { cameras: availableCameras } = useAvailableCameras({ enabled });

  // Bind each expected camera that has no camera yet: to the one it had on the
  // last run, else to a robot camera of the same name. Prefer matching by
  // browser device_id (stable across cv2 index drift); fall back to the index.
  const rememberedCameras = remembered.cameras;
  useEffect(() => {
    if (!imageFeatures || availableCameras.length === 0) return;
    const robotCams = robot?.cameras ?? [];
    setBindings((prev) => {
      let changed = false;
      const next = { ...prev };
      // A camera already bound to another name isn't bound twice; that name is left to the user.
      const used = new Set(Object.values(next).filter((index) => index != null));
      for (const expectedName of Object.keys(imageFeatures)) {
        if (next[expectedName] != null) continue;
        const last = rememberedCameras[expectedName];
        const robotCam = robotCams.find(
          (c) => c.name.toLowerCase() === expectedName.toLowerCase(),
        );
        const live = [
          last ? liveCameraFor(last, availableCameras) : undefined,
          robotCam ? liveCameraFor(robotCam, availableCameras) : undefined,
        ].find((camera) => camera && !used.has(camera.index));
        if (live) {
          next[expectedName] = live.index;
          used.add(live.index);
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [imageFeatures, robot, availableCameras, rememberedCameras]);

  const allBound =
    !!imageFeatures && Object.keys(imageFeatures).every((name) => bindings[name] != null);

  const bind = (name: string, index: number) =>
    setBindings((prev) => ({ ...prev, [name]: index }));

  // Rotations picked on the last run come back with the dialog.
  const rememberedRotations = remembered.rotations;
  useEffect(() => {
    if (enabled) setRotations(rememberedRotations);
  }, [enabled, robot?.name, rememberedRotations]);

  const physicalFor = (index: number): PhysicalCamera =>
    availableCameras.find((c) => c.index === index) ?? { index, deviceId: "" };

  const rotationFor = (name: string): CameraRotation => {
    const idx = bindings[name];
    return idx == null
      ? 0
      : rotationForCamera(physicalFor(idx), robot?.cameras ?? [], rotations);
  };

  const setRotation = (name: string, rotation: CameraRotation) => {
    const idx = bindings[name];
    if (idx != null) {
      setRotations((prev) => ({ ...prev, [physicalCameraKey(physicalFor(idx))]: rotation }));
    }
  };

  /**
   * The backend camera config: every bound camera. The expected resolution is
   * the rotated output, so the API gets the capture dimensions.
   */
  const toCameraDict = (fps: number) => {
    const cameraDict: Record<string, CameraRequest> = {};
    for (const [name, dims] of Object.entries(imageFeatures ?? {})) {
      const idx = bindings[name];
      if (idx == null) continue;
      const rotation = rotationFor(name);
      const capture = rotatedDimensions(dims.width, dims.height, rotation);
      cameraDict[name] = serializeCamera({
        type: "opencv",
        camera_index: idx,
        width: capture.width,
        height: capture.height,
        rotation,
        fps,
      });
    }
    return cameraDict;
  };

  /** The bindings and rotations to remember for the next run. */
  const remember = (): RememberedBindings => {
    const cameras: Record<string, RememberedCamera> = {};
    for (const name of Object.keys(imageFeatures ?? {})) {
      const idx = bindings[name];
      if (idx == null) continue;
      const physical = physicalFor(idx);
      cameras[name] = { device_id: physical.deviceId, camera_index: physical.index };
    }
    return { cameras, rotations };
  };

  return { bindings, bind, allBound, availableCameras, rotationFor, setRotation, toCameraDict, remember };
}
