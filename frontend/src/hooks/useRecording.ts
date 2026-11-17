import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { RobotRecord } from "@/hooks/useRobots";
import { CameraConfig, cameraConfigurationError, serializeCameras } from "@/lib/cameraConfig";
import { useCameraBindings } from "@/hooks/useCameraBindings";
import { ImageFeatures } from "@/lib/checkpointsApi";
import { useOptionalRemoteServers } from "@/contexts/RemoteServerContext";

const SEND_AFTER_KEY = "lelab.sendToServerAfterRecording";

const readSendAfter = (): boolean => {
  try {
    return window.localStorage.getItem(SEND_AFTER_KEY) === "1";
  } catch {
    return false;
  }
};

/** An existing local dataset that new episodes get appended to. */
export interface AppendTarget {
  repoId: string;
  numEpisodes: number;
  /** Episodes the recording was meant to have; null when LeLab doesn't know. */
  targetEpisodes: number | null;
  fps: number;
  imageFeatures: ImageFeatures;
}

/** Whether the dataset stopped short of the episodes it was meant to have. */
export const isUnfinished = (target: AppendTarget): boolean =>
  target.targetEpisodes !== null && target.targetEpisodes > target.numEpisodes;

/** Episodes appending starts with: the rest of the plan, else 5 more. */
export const defaultTotalEpisodes = (target: AppendTarget): number =>
  isUnfinished(target) ? (target.targetEpisodes as number) : target.numEpisodes + 5;

/**
 * Owns the recording modal's state and the "start recording" orchestration,
 * shared by the Landing page (new dataset) and the dataset browser (append
 * episodes to the dataset being browsed).
 */
export const useRecording = (robot: RobotRecord | null) => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const { auth } = useHfAuth();

  const [open, setOpen] = useState(false);
  const [appendTo, setAppendTo] = useState<AppendTarget | null>(null);
  const [datasetName, setDatasetName] = useState("");
  const [singleTask, setSingleTask] = useState("");
  const [numEpisodes, setNumEpisodes] = useState(5);
  // Appending asks for the dataset's new total rather than a count to add.
  const [totalEpisodes, setTotalEpisodes] = useState(0);
  const [episodeTimeS, setEpisodeTimeS] = useState(60);
  const [resetTimeS, setResetTimeS] = useState(15);
  const [streamingEncoding, setStreamingEncoding] = useState(true);
  const [cameras, setCameras] = useState<CameraConfig[]>([]);
  const remote = useOptionalRemoteServers();
  const [sendToServer, setSendToServerState] = useState<boolean>(readSendAfter);
  const setSendToServer = (value: boolean) => {
    setSendToServerState(value);
    try {
      window.localStorage.setItem(SEND_AFTER_KEY, value ? "1" : "0");
    } catch {
      // storage unavailable: the choice lasts for this page only
    }
  };
  // Only offered (and only acted on) when a server is saved.
  const sendServer = remote?.selected ?? null;

  const releaseStreamsRef = useRef<(() => void) | null>(null);
  // Appending must feed every camera the dataset has, at its resolution, so
  // the robot's cameras are bound to those slots the way inference binds them.
  const cameraBindings = useCameraBindings(
    appendTo?.imageFeatures ?? null,
    robot,
    open && appendTo !== null,
  );

  /** Open the modal to record a brand-new dataset with the robot's cameras. */
  const openForNew = (name: string) => {
    setDatasetName(name);
    setCameras(robot ? [...(robot.cameras ?? [])] : []);
    setOpen(true);
  };

  /** Open the modal to append episodes, prefilled with the dataset's last task. */
  const openToAppend = (target: AppendTarget, lastTask: string) => {
    setAppendTo(target);
    setTotalEpisodes(defaultTotalEpisodes(target));
    setSingleTask(lastTask);
    setOpen(true);
  };

  // A bound camera keeps the capture settings the robot gives that device, as
  // in a new recording.
  const captureSettings = (index: number | undefined) => {
    const robotCams = robot?.cameras ?? [];
    const deviceId = cameraBindings.availableCameras.find((c) => c.index === index)?.deviceId;
    const cam =
      (deviceId && robotCams.find((c) => c.device_id === deviceId)) ||
      robotCams.find((c) => c.camera_index === index);
    return {
      ...(cam?.fourcc ? { fourcc: cam.fourcc } : {}),
      ...(cam?.backend ? { backend: cam.backend } : {}),
    };
  };

  const onStart = async () => {
    // Start stays disabled until a calibrated robot is selected.
    if (!robot) return;
    const cameraError = cameraConfigurationError(robot.cameras ?? []);
    if (cameraError) {
      toast({
        title: "Invalid camera rotation",
        description: `${cameraError} Correct the camera settings in Calibration.`,
        variant: "destructive",
      });
      return;
    }
    if ((!appendTo && !datasetName) || !singleTask) {
      toast({
        title: "Missing dataset details",
        description: appendTo
          ? "Please enter a task description."
          : "Please enter a dataset name and task description.",
        variant: "destructive",
      });
      return;
    }
    if (appendTo && totalEpisodes <= appendTo.numEpisodes) {
      toast({
        title: "Total episodes too low",
        description: `The dataset already has ${appendTo.numEpisodes} episodes. Enter a higher total.`,
        variant: "destructive",
      });
      return;
    }

    // An appended dataset keeps its exact on-disk id: the backend only stamps
    // a timestamp onto new recordings.
    const datasetRepoId = appendTo
      ? appendTo.repoId
      : auth.status === "authenticated"
        ? `${auth.username}/${datasetName}`
        : datasetName;

    if (cameras.length > 0 && releaseStreamsRef.current) {
      console.log("🔓 Releasing camera streams before starting recording...");
      toast({
        title: "Preparing Camera Resources",
        description: `Releasing ${cameras.length} camera stream(s) for recording...`,
      });
      releaseStreamsRef.current();
      await new Promise((resolve) => setTimeout(resolve, 500));
      console.log("✅ Camera streams released, proceeding with recording...");
      toast({
        title: "Camera Resources Ready",
        description:
          "Camera streams released successfully. Starting recording...",
      });
    }

    const cameraDict = appendTo
      ? Object.fromEntries(
          Object.entries(cameraBindings.toCameraDict(appendTo.fps)).map(([name, cam]) => [
            name,
            { ...cam, ...captureSettings(cam.camera_index) },
          ]),
        )
      : serializeCameras(cameras);

    const recordingConfig = {
      leader_port: robot.leader_port,
      follower_port: robot.follower_port,
      leader_config: robot.leader_config,
      follower_config: robot.follower_config,
      dataset_repo_id: datasetRepoId,
      single_task: singleTask,
      // The backend records this many in the session; on append that is
      // whatever the new total still needs.
      num_episodes: appendTo ? totalEpisodes - appendTo.numEpisodes : numEpisodes,
      episode_time_s: episodeTimeS,
      reset_time_s: resetTimeS,
      // LeRobot refuses to append frames at a different rate than the dataset's.
      fps: appendTo ? appendTo.fps : 30,
      video: true,
      push_to_hub: false,
      resume: appendTo !== null,
      streaming_encoding: streamingEncoding,
      cameras: cameraDict,
    };

    setOpen(false);
    navigate("/recording", {
      state: { recordingConfig, sendToServer: sendToServer && sendServer !== null },
    });
  };

  return {
    openForNew,
    openToAppend,
    modalProps: {
      open,
      onOpenChange: setOpen,
      robot,
      appendTo,
      cameraBindings,
      datasetName,
      setDatasetName,
      singleTask,
      setSingleTask,
      numEpisodes,
      setNumEpisodes,
      totalEpisodes,
      setTotalEpisodes,
      episodeTimeS,
      setEpisodeTimeS,
      resetTimeS,
      setResetTimeS,
      streamingEncoding,
      setStreamingEncoding,
      cameras,
      setCameras,
      onStart,
      releaseStreamsRef,
      sendServerName: sendServer?.name ?? null,
      sendToServer,
      setSendToServer,
    },
  };
};
