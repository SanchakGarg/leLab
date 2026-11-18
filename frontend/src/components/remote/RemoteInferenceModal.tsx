import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AlertTriangle, CheckCircle, ChevronDown, Loader2, Play, PlugZap, Server } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { NumberInput } from "@/components/ui/number-input";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import CameraBindings from "@/components/landing/CameraBindings";
import ChunkBlendingOption from "@/components/inference/ChunkBlendingOption";
import InteractiveOptions from "@/components/inference/InteractiveOptions";
import { useApi } from "@/contexts/ApiContext";
import { useRemoteServers } from "@/contexts/RemoteServerContext";
import { useCameraBindings } from "@/hooks/useCameraBindings";
import {
  REMOTE_RUN_SETTINGS_KEY,
  blendingRequest,
  interactiveRequest,
  useRunSettings,
} from "@/hooks/useRunSettings";
import { useRunVideo } from "@/hooks/useRunVideo";
import { useToast } from "@/hooks/use-toast";
import { RobotRecord } from "@/hooks/useRobots";
import { cameraConfigurationError } from "@/lib/cameraConfig";
import {
  PolicyConnectionKind,
  PolicyHello,
  errorText,
  imageFeaturesFromHello,
  probePolicyServer,
  startRemoteInference,
} from "@/lib/remoteApi";
import RemoteServerDialog from "./RemoteServerDialog";
import RunVideoOptions from "./RunVideoOptions";
import ServerConnectForm from "./ServerConnectForm";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  robot: RobotRecord | null;
}

const LAST_KEY = "lelab.remotePolicy";

interface LastTarget {
  connection: PolicyConnectionKind;
  host: string;
  port: number;
}

const readLast = (): LastTarget => {
  try {
    const saved = JSON.parse(window.localStorage.getItem(LAST_KEY) ?? "null");
    if (saved && typeof saved.host === "string" && typeof saved.port === "number") return saved;
  } catch {
    // ignore: fall back to the defaults
  }
  return { connection: "tunnel", host: "", port: 5555 };
};

/**
 * Run a model that is served on the remote server (lelab_policy_server.py) on this arm:
 * connect to it (through the server's SSH session or directly), read which cameras it
 * needs, bind them, and start. The control loop runs here; the model runs there.
 */
const RemoteInferenceModal: React.FC<Props> = ({ open, onOpenChange, robot }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const navigate = useNavigate();
  const { servers, selected, selectServer } = useRemoteServers();

  const [connection, setConnection] = useState<PolicyConnectionKind>("tunnel");
  const [host, setHost] = useState("");
  const [port, setPort] = useState(5555);
  const [hello, setHello] = useState<PolicyHello | null>(null);
  const [probing, setProbing] = useState(false);
  const [probeError, setProbeError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  // Prefilled from the last run on a server each time the dialog opens; saved when a run starts.
  const { settings, update, save } = useRunSettings(REMOTE_RUN_SETTINGS_KEY, open);

  useEffect(() => {
    if (!open) return;
    const last = readLast();
    setConnection(last.connection);
    setHost(last.host);
    setPort(last.port);
    setHello(null);
    setProbeError(null);
    setSubmitting(false);
  }, [open]);

  const imageFeatures = useMemo(
    () => (hello ? imageFeaturesFromHello(hello, robot?.cameras ?? []) : null),
    [hello, robot?.cameras],
  );
  const cameraBindings = useCameraBindings(imageFeatures, robot, open && !!hello, settings);
  const cameraError = robot ? cameraConfigurationError(robot.cameras ?? []) : null;
  const policyCameras = hello?.cameras ?? [];
  const boundName = (camera: string) => {
    const index = cameraBindings.bindings[camera];
    return index == null ? null : robot?.cameras?.find((c) => c.camera_index === index)?.name;
  };
  const video = useRunVideo(policyCameras, boundName, settings.videoChoice);
  const { choose: chooseVideo } = video;
  useEffect(() => {
    if (open) chooseVideo(null); // back to the remembered pick each time the dialog opens
  }, [open, chooseVideo]);
  const isInteractive = settings.interactive;
  const tunnel = connection === "tunnel";
  const serverReady = !tunnel || !!selected?.session.connected;
  const target = {
    server: tunnel ? selected?.name ?? null : null,
    connection,
    policy_host: host.trim() || (tunnel ? "127.0.0.1" : ""),
    policy_port: port,
  };

  const probe = async () => {
    setProbing(true);
    setProbeError(null);
    setHello(null);
    try {
      const result = await probePolicyServer(baseUrl, fetchWithHeaders, target);
      setHello(result);
      try {
        window.localStorage.setItem(LAST_KEY, JSON.stringify({ connection, host: host.trim(), port }));
      } catch {
        // storage unavailable
      }
    } catch (e) {
      setProbeError(errorText(e));
    } finally {
      setProbing(false);
    }
  };

  const fps = hello?.fps || 30;
  const canStart =
    !!robot && robot.is_clean && !cameraError && !!hello && serverReady && cameraBindings.allBound && !submitting;

  const handleStart = async () => {
    if (!robot || !hello) return;
    // Previews drop their browser streams so the backend can open the same devices.
    setSubmitting(true);
    await new Promise((r) => setTimeout(r, 300));
    try {
      await startRemoteInference(baseUrl, fetchWithHeaders, {
        ...target,
        follower_port: robot.follower_port,
        follower_config: robot.follower_config,
        task: isInteractive ? "" : settings.task,
        cameras: cameraBindings.toCameraDict(fps),
        duration_s: settings.durationS,
        fps,
        n_action_steps: settings.nActionSteps,
        prefetch: settings.prefetch,
        max_relative_target: settings.maxStep,
        ...video.request,
        ...interactiveRequest(settings),
        ...blendingRequest(settings),
      });
      save({ ...cameraBindings.remember(), videoChoice: video.choice });
      onOpenChange(false);
      navigate("/inference");
    } catch (e) {
      toast({ title: "Couldn't start the remote model", description: errorText(e), variant: "destructive" });
      setSubmitting(false);
    }
  };

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="bg-gray-900 border-gray-800 text-white sm:max-w-[620px] p-8 max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center justify-center gap-2 text-white text-2xl font-bold">
              <Server className="h-6 w-6" />
              Run a model on your server
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-6 py-2">
            <DialogDescription className="text-gray-400 text-center">
              Start <code>lelab_policy_server.py</code> on the server with your trained checkpoint, then enter the
              IP and port its log prints. The model runs there; this arm follows its actions.
            </DialogDescription>

            <section className="space-y-3">
              <h3 className="text-lg font-semibold text-white border-b border-gray-700 pb-2">Robot</h3>
              {!robot ? (
                <Alert className="bg-amber-900/40 border-amber-700 text-amber-100">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>Select and configure a robot on the Landing page first.</AlertDescription>
                </Alert>
              ) : !robot.is_clean ? (
                <Alert className="bg-amber-900/40 border-amber-700 text-amber-100">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    <strong>{robot.name}</strong> is missing a calibration. Configure it first.
                  </AlertDescription>
                </Alert>
              ) : (
                <div className="flex items-center gap-2 text-sm">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span className="text-slate-200">
                    Running on <strong>{robot.name}</strong>
                  </span>
                </div>
              )}
            </section>

            <section className="space-y-3">
              <h3 className="text-lg font-semibold text-white border-b border-gray-700 pb-2">Policy server</h3>
              <div className="space-y-1.5">
                <Label className="text-sm text-gray-300">Connect</Label>
                <Select
                  value={connection}
                  onValueChange={(v) => {
                    setConnection(v as PolicyConnectionKind);
                    setHello(null);
                  }}
                >
                  <SelectTrigger className="bg-gray-800 border-gray-700 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-gray-900 border-gray-700 text-white">
                    <SelectItem value="tunnel">Through the server's SSH login (tunnel)</SelectItem>
                    <SelectItem value="direct">Directly (same network)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {tunnel ? (
                !selected ? (
                  <Button onClick={() => setSettingsOpen(true)} className="bg-blue-600 text-white hover:bg-blue-500">
                    Add a server
                  </Button>
                ) : (
                  <div className="space-y-3">
                    {servers.length > 1 ? (
                      <Select value={selected.name} onValueChange={selectServer}>
                        <SelectTrigger className="bg-gray-800 border-gray-700 text-white">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-gray-900 border-gray-700 text-white">
                          {servers.map((s) => (
                            <SelectItem key={s.name} value={s.name}>
                              {s.name} — {s.username}@{s.host}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    ) : null}
                    <ServerConnectForm server={selected} />
                  </div>
                )
              ) : null}

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                <div className="space-y-1.5 sm:col-span-2">
                  <Label htmlFor="policy-host" className="text-sm text-gray-300">
                    {tunnel ? "GPU node IP (as the server sees it)" : "Host"}
                  </Label>
                  <Input
                    id="policy-host"
                    value={host}
                    onChange={(e) => {
                      setHost(e.target.value);
                      setHello(null);
                    }}
                    placeholder={tunnel ? "127.0.0.1 when it runs on the login host" : "10.2.0.7"}
                    className="bg-gray-800 border-gray-700 text-white"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="policy-port" className="text-sm text-gray-300">Port</Label>
                  <NumberInput
                    id="policy-port"
                    min={1}
                    max={65535}
                    value={port}
                    onChange={(v) => {
                      if (v !== undefined) setPort(v);
                      setHello(null);
                    }}
                    className="bg-gray-800 border-gray-700 text-white"
                  />
                </div>
              </div>
              <Button
                onClick={probe}
                disabled={probing || !serverReady || (!tunnel && !host.trim())}
                variant="outline"
                className="border-gray-700 bg-gray-950 text-gray-200 hover:bg-gray-900"
              >
                {probing ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <PlugZap className="mr-2 h-4 w-4" />}
                Connect to the model
              </Button>
              {probeError ? <p className="text-sm text-red-300">{probeError}</p> : null}
              {hello ? (
                <div className="rounded-md border border-emerald-800/60 bg-emerald-950/30 px-3 py-2 text-sm text-emerald-100">
                  <p className="break-all">
                    <strong>{hello.name}</strong>
                  </p>
                  <p className="text-xs text-emerald-200/80">
                    {hello.address} · cameras {hello.cameras.join(", ") || "none"} · chunk of {hello.chunk_size}{" "}
                    actions, plays {hello.n_action_steps} at {hello.fps} Hz
                  </p>
                </div>
              ) : null}
            </section>

            {hello ? (
              <>
                <section className="space-y-3">
                  <h3 className="text-lg font-semibold text-white border-b border-gray-700 pb-2">Cameras</h3>
                  {cameraError ? (
                    <Alert className="bg-red-900/40 border-red-700 text-red-100">
                      <AlertTriangle className="h-4 w-4" />
                      <AlertDescription>{cameraError} Correct the camera settings in Calibration.</AlertDescription>
                    </Alert>
                  ) : imageFeatures && Object.keys(imageFeatures).length > 0 ? (
                    <div className="space-y-3">
                      <p className="text-xs text-gray-500">
                        Bind a camera to each name the model was trained with, with the same rotation as the
                        training data.
                      </p>
                      <CameraBindings
                        imageFeatures={imageFeatures}
                        bindings={cameraBindings.bindings}
                        onBind={cameraBindings.bind}
                        rotationFor={cameraBindings.rotationFor}
                        onRotate={cameraBindings.setRotation}
                        availableCameras={cameraBindings.availableCameras}
                        paused={submitting}
                      />
                    </div>
                  ) : (
                    <p className="text-xs text-gray-500">This model doesn't use cameras.</p>
                  )}
                </section>

                <section className="space-y-3">
                  <h3 className="text-lg font-semibold text-white border-b border-gray-700 pb-2">Run</h3>
                  <InteractiveOptions idPrefix="remote" settings={settings} onChange={update} />
                  {isInteractive ? null : (
                    <>
                      <div className="space-y-1.5">
                        <Label htmlFor="remote-task" className="text-sm text-gray-300">Task (for language-conditioned models)</Label>
                        <Input
                          id="remote-task"
                          value={settings.task}
                          onChange={(e) => update({ task: e.target.value })}
                          placeholder="e.g., pick up the red block"
                          className="bg-gray-800 border-gray-700 text-white"
                        />
                      </div>
                      <div className="space-y-1.5">
                        <Label htmlFor="remote-duration" className="text-sm text-gray-300">Max duration (seconds)</Label>
                        <NumberInput
                          id="remote-duration"
                          min={1}
                          value={settings.durationS}
                          onChange={(v) => {
                            if (v !== undefined) update({ durationS: v });
                          }}
                          className="bg-gray-800 border-gray-700 text-white"
                        />
                      </div>
                    </>
                  )}
                  <ChunkBlendingOption
                    id="remote-chunk-blending"
                    settings={settings}
                    onChange={update}
                    fps={fps}
                    note={isInteractive ? "You can also switch it between commands on the session page." : null}
                    warning={
                      settings.prefetch === 0
                        ? "Prefetch (under Advanced) is 0: the arm plays every chunk to its end before the next one comes, so there is nothing to blend."
                        : settings.blendSteps > settings.prefetch
                          ? `At most ${settings.prefetch} actions are left when a new chunk arrives (Prefetch, under Advanced), so at most that many get blended.`
                          : null
                    }
                  />
                  <RunVideoOptions id="remote-record-video" video={video} what={isInteractive ? "session" : "run"} />
                  <Collapsible className="group space-y-3">
                    <CollapsibleTrigger className="flex w-full items-center justify-between text-sm font-medium text-gray-300">
                      Advanced
                      <ChevronDown className="h-4 w-4 transition-transform group-data-[state=open]:rotate-180" />
                    </CollapsibleTrigger>
                    <CollapsibleContent className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                      <div className="space-y-1.5">
                        <Label htmlFor="remote-maxstep" className="text-xs text-gray-400">Max move per step</Label>
                        <NumberInput
                          id="remote-maxstep"
                          min={1}
                          value={settings.maxStep}
                          onChange={(v) => update({ maxStep: v ?? null })}
                          className="bg-gray-800 border-gray-700 text-white"
                        />
                      </div>
                      <div className="space-y-1.5">
                        <Label htmlFor="remote-nsteps" className="text-xs text-gray-400">Actions per chunk</Label>
                        <NumberInput
                          id="remote-nsteps"
                          min={1}
                          max={hello.chunk_size}
                          value={settings.nActionSteps}
                          onChange={(v) => update({ nActionSteps: v ?? null })}
                          placeholder={String(hello.n_action_steps)}
                          className="bg-gray-800 border-gray-700 text-white"
                        />
                      </div>
                      <div className="space-y-1.5">
                        <Label htmlFor="remote-prefetch" className="text-xs text-gray-400">Prefetch (actions left)</Label>
                        <NumberInput
                          id="remote-prefetch"
                          min={0}
                          value={settings.prefetch}
                          onChange={(v) => update({ prefetch: v ?? 0 })}
                          className="bg-gray-800 border-gray-700 text-white"
                        />
                      </div>
                      <p className="text-xs text-gray-500 sm:col-span-3">
                        Max move caps each step in degrees (gripper 0–100); leave empty for no cap. Fewer actions
                        per chunk replans more often. The next chunk is requested while this many actions are left.
                      </p>
                    </CollapsibleContent>
                  </Collapsible>
                </section>
              </>
            ) : null}

            <div className="flex flex-col sm:flex-row gap-4 justify-center pt-2">
              <Button
                onClick={handleStart}
                disabled={!canStart}
                className="w-full sm:w-auto bg-green-500 hover:bg-green-600 text-white px-10 py-6 text-lg disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <Play className="w-5 h-5 mr-2" />
                {submitting ? "Starting…" : isInteractive ? "Start session" : "Start"}
              </Button>
              <Button
                onClick={() => onOpenChange(false)}
                variant="outline"
                className="w-full sm:w-auto border-gray-500 hover:border-gray-200 px-10 py-6 text-lg text-zinc-500 bg-zinc-900 hover:bg-zinc-800"
              >
                Cancel
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
      <RemoteServerDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
    </>
  );
};

export default RemoteInferenceModal;
