import React, { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Download, Loader2, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import Logo from "@/components/Logo";
import ArmLinkNotice from "@/components/control/ArmLinkNotice";
import CommandHistory from "@/components/inference/CommandHistory";
import CommandInput from "@/components/inference/CommandInput";
import { SessionBlendingToggle } from "@/components/inference/ChunkBlendingOption";
import { runVideoUrl } from "@/lib/runVideo";
import { errorText } from "@/lib/remoteApi";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  InferenceStatus,
  InteractiveCommand,
  InteractiveStatus,
  endInferenceCommand,
  getInferenceStatus,
  sendInferenceCommand,
  setChunkBlending,
  stopInference,
  summarizeCommands,
} from "@/lib/inferenceApi";

const POLL_MS = 1000;
// How often a running command's clock moves between status polls.
const TICK_MS = 250;

function formatTime(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(s / 60);
  const secs = s % 60;
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function blockedReason(status: InferenceStatus, session: InteractiveStatus): string | null {
  if (!status.inference_active) return "The session has ended.";
  if (session.state === "setting up" || status.rollout_started_at == null)
    return "The model is loading. You can type now and send once it's ready.";
  if (session.state === "returning") return "The arm is going back to its start pose.";
  if (session.state === "running") return "A command is running. Press Done or Stop command to end it early.";
  return null;
}

const Inference: React.FC = () => {
  const navigate = useNavigate();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const [status, setStatus] = useState<InferenceStatus | null>(null);
  const [showStopConfirm, setShowStopConfirm] = useState(false);
  const [ending, setEnding] = useState(false);
  // Chunk blending as just clicked, until the status reports it.
  const [blendPending, setBlendPending] = useState<boolean | null>(null);
  // A finished run with a video or a command history: stay here to show it instead of going home.
  const [finished, setFinished] = useState<InferenceStatus | null>(null);
  const navigatedAwayRef = useRef(false);
  // Independent flag: we may request a stop (safety net) before the run
  // is actually inactive. We must not flip navigatedAwayRef yet — that
  // would block the natural completion path on the next tick.
  const stopRequestedRef = useRef(false);
  // When the last status arrived, to move a running command's clock between polls.
  const receivedAtRef = useRef(0);
  const refreshRef = useRef<() => void>(() => {});
  const [, setTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    // One status request at a time: a finished run is reported once, so two overlapping requests
    // could have the second (idle) one land first and lose the result.
    let busy = false;
    let again = false;
    const stopIfHung = async () => {
      try {
        await stopInference(baseUrl, fetchWithHeaders);
      } catch {
        // The next status poll will surface the failure if it persists.
      }
    };
    const poll = async () => {
      try {
        const next = await getInferenceStatus(baseUrl, fetchWithHeaders);
        if (cancelled || navigatedAwayRef.current) return;
        receivedAtRef.current = performance.now();
        setStatus(next);
        // Auto-bounce home once the run is done.
        if (!next.inference_active && !navigatedAwayRef.current) {
          navigatedAwayRef.current = true;
          const commands = next.interactive?.commands ?? [];
          if (next.exited) {
            const failed = next.exit_code !== 0;
            const videoError = next.video?.error ? ` The video could not be saved: ${next.video.error}` : "";
            const session = next.interactive != null;
            toast({
              title: failed
                ? session
                  ? "The session stopped with an error"
                  : "Inference stopped with an error"
                : session
                  ? "Session ended"
                  : "Inference finished",
              description:
                (!failed
                  ? session
                    ? summarizeCommands(commands)
                    : "Run completed."
                  : next.hint || next.error || (next.log_path ? `Exit code ${next.exit_code}. See ${next.log_path}.` : `Exit code ${next.exit_code}.`)) +
                videoError,
              variant: failed ? "destructive" : "default",
            });
          }
          if (next.video?.name || commands.length > 0) {
            setFinished(next);
            return;
          }
          navigate("/");
          return;
        }
        // Safety net: only fire after the rollout *main loop* has actually
        // started (lerobot honours --duration there). Setup time — policy
        // load, snapshot_download, bus connect, camera connect — can take
        // 10–30s and must NOT count against the user's configured duration.
        // Interactive sessions report no duration (each command has its own limit).
        if (
          next.inference_active &&
          next.rollout_started_at != null &&
          next.duration_s != null &&
          next.duration_s > 0 &&
          next.rollout_elapsed_s > next.duration_s + 10 &&
          !stopRequestedRef.current
        ) {
          stopRequestedRef.current = true;
          toast({
            title: "Inference seems hung",
            description: `Rollout past duration by ${Math.round(
              next.rollout_elapsed_s - next.duration_s,
            )}s. Stopping.`,
            variant: "destructive",
          });
          stopIfHung();
        }
      } catch (e) {
        if (!cancelled) {
          toast({
            title: "Lost connection to backend",
            description: e instanceof Error ? e.message : String(e),
            variant: "destructive",
          });
        }
      }
    };
    const tick = async () => {
      if (busy) {
        again = true;
        return;
      }
      busy = true;
      try {
        await poll();
      } finally {
        busy = false;
        if (again && !cancelled) {
          again = false;
          void tick();
        }
      }
    };
    refreshRef.current = () => void tick();
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
      refreshRef.current = () => {};
    };
  }, [baseUrl, fetchWithHeaders, navigate, toast]);

  const session = status?.inference_active ? status.interactive ?? null : null;
  const commandRunning = session?.state === "running";
  const sessionBlending = session?.chunk_blending;
  useEffect(() => {
    setBlendPending(null);
  }, [sessionBlending]);
  // Move a running command's clock between polls.
  useEffect(() => {
    if (!commandRunning) return;
    const id = setInterval(() => setTick((n) => n + 1), TICK_MS);
    return () => clearInterval(id);
  }, [commandRunning]);

  const elapsed = useCallback(
    (command: InteractiveCommand) => {
      if (command.status !== "running") return command.elapsed_s;
      const since = (performance.now() - receivedAtRef.current) / 1000;
      return Math.min(command.elapsed_s + Math.max(0, since), session?.command_time_s ?? Infinity);
    },
    [session?.command_time_s],
  );

  const handleStop = async () => {
    setShowStopConfirm(false);
    try {
      await stopInference(baseUrl, fetchWithHeaders);
      // Status poll will catch the inactive state and navigate home.
    } catch (e) {
      toast({
        title: "Stop failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  const handleSend = async (text: string) => {
    try {
      await sendInferenceCommand(baseUrl, fetchWithHeaders, text);
      refreshRef.current();
      return true;
    } catch (e) {
      toast({ title: "Couldn't send the command", description: errorText(e), variant: "destructive" });
      return false;
    }
  };

  const handleBlending = async (enabled: boolean) => {
    setBlendPending(enabled);
    try {
      await setChunkBlending(baseUrl, fetchWithHeaders, enabled);
      refreshRef.current();
    } catch (e) {
      setBlendPending(null);
      toast({ title: "Couldn't change chunk blending", description: errorText(e), variant: "destructive" });
    }
  };

  const handleEnd = async (completed: boolean) => {
    setEnding(true);
    try {
      await endInferenceCommand(baseUrl, fetchWithHeaders, completed);
      refreshRef.current();
    } catch (e) {
      toast({
        title: completed ? "Couldn't mark the command done" : "Couldn't stop the command",
        description: errorText(e),
        variant: "destructive",
      });
    } finally {
      setEnding(false);
    }
  };

  if (finished) {
    const video = finished.video?.name ? finished.video : null;
    const url = video ? runVideoUrl(baseUrl, video.name!) : null;
    const commands = finished.interactive?.commands ?? [];
    return (
      <div className="min-h-screen bg-black text-white flex flex-col items-center justify-center gap-5 p-4 sm:p-6 lg:p-8">
        <div className="w-full max-w-3xl space-y-4 rounded-lg border border-gray-700 bg-gray-900 p-6">
          <div className="flex items-center justify-between gap-4">
            <h1 className="shrink-0 text-xl font-bold">
              {finished.interactive ? (video ? "Session video" : "Session") : "Run video"}
            </h1>
            <span className="min-w-0 text-xs text-gray-500 break-all text-right">
              {finished.policy_ref ?? ""}
            </span>
          </div>
          {video && url ? (
            <>
              <video src={url} controls autoPlay muted playsInline className="w-full rounded-md bg-black" />
              <p className="text-xs text-gray-500 break-all">
                {video.camera}
                {video.corner_camera ? `, ${video.corner_camera} in the corner` : ""} · {video.frames} frames
                {video.dropped ? ` (${video.dropped} dropped)` : ""} · saved at {video.path}
              </p>
            </>
          ) : null}
          {finished.interactive ? (
            <div className="space-y-2">
              <div className="flex items-baseline justify-between gap-3">
                <h2 className="text-sm font-semibold text-gray-200">Commands</h2>
                <span className="text-xs text-gray-500">{summarizeCommands(commands)}</span>
              </div>
              <CommandHistory commands={commands} commandTimeS={finished.interactive.command_time_s} />
            </div>
          ) : null}
          <div className="flex flex-col gap-3 sm:flex-row sm:justify-end">
            {video && url ? (
              <Button asChild variant="outline" className="border-gray-600 bg-gray-950 text-gray-200 hover:bg-gray-800">
                <a href={url} download={video.name ?? undefined}>
                  <Download className="mr-2 h-4 w-4" />
                  Download
                </a>
              </Button>
            ) : null}
            <Button onClick={() => navigate("/")} className="bg-white text-black hover:bg-gray-200">
              Back to home
            </Button>
          </div>
        </div>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin mr-3" /> Connecting to inference…
      </div>
    );
  }

  const setupElapsed = status.elapsed_s ?? 0;
  const rolloutElapsed = status.rollout_elapsed_s ?? 0;
  const duration = status.duration_s ?? 0;
  const isSettingUp = status.inference_active && status.rollout_started_at == null;
  const isRunning = status.inference_active && status.rollout_started_at != null;
  // When setting up: progress is uncertain — show a soft pulsing bar.
  // When rolling out: progress is rolloutElapsed / duration.
  const pct =
    isRunning && duration > 0
      ? Math.min(100, (rolloutElapsed / duration) * 100)
      : 0;
  const pillLabel = isSettingUp
    ? "SETTING UP"
    : isRunning
    ? "RUNNING"
    : "FINISHED";
  const timerSeconds = isRunning ? rolloutElapsed : setupElapsed;

  const policyLine = (
    <div className="text-xs text-slate-500 break-all">
      policy: {status.policy_ref ?? "(unknown)"}
      {status.remote ? (
        <div className="mt-1 text-slate-400">
          {status.mode === "local" ? "local model" : "remote"}: {status.remote.phase} · {status.remote.chunks} chunks · round trip{" "}
          {status.remote.rtt_ms_median != null ? `${Math.round(status.remote.rtt_ms_median)} ms` : "–"} (inference{" "}
          {status.remote.infer_ms_median != null ? `${Math.round(status.remote.infer_ms_median)} ms` : "–"}) · waited{" "}
          {status.remote.wait_s.toFixed(1)} s
          {status.remote.arm_skipped ? ` · ${status.remote.arm_skipped} arm reads/writes skipped` : ""}
          {status.remote.blend_steps
            ? ` · blending over ${status.remote.blend_steps} actions (${status.remote.blended ?? 0} blended)`
            : ""}
        </div>
      ) : null}
    </div>
  );

  const header = (
    <div className="flex items-center gap-4 mb-8">
      <Button
        variant="ghost"
        size="icon"
        onClick={() => navigate("/")}
        className="text-slate-400 hover:bg-slate-800 hover:text-white rounded-lg"
      >
        <ArrowLeft className="w-5 h-5" />
      </Button>
      <Logo />
      <h1 className="font-bold text-white text-2xl">Inference</h1>
    </div>
  );

  const stopDialog = (
    <AlertDialog open={showStopConfirm} onOpenChange={setShowStopConfirm}>
      <AlertDialogContent className="bg-gray-900 border-gray-700 text-white">
        <AlertDialogHeader>
          <AlertDialogTitle>{session ? "End the session?" : "Stop inference?"}</AlertDialogTitle>
          <AlertDialogDescription className="text-gray-400">
            {session
              ? "A command still running is stopped. The follower returns to the pose it started from, then its motors are released."
              : "The follower returns to the pose it started from, then its motors are released. You can launch another run from the jobs list."}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel className="bg-gray-800 border-gray-700 text-white hover:bg-gray-700">
            {session ? "Keep going" : "Keep running"}
          </AlertDialogCancel>
          <AlertDialogAction
            onClick={handleStop}
            className="bg-red-500 hover:bg-red-600 text-white"
          >
            {session ? "End session" : "Stop"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );

  if (session) {
    const current = session.commands.find((c) => c.status === "pending" || c.status === "running") ?? null;
    const settingUp = session.state === "setting up" || isSettingUp;
    const pill = settingUp
      ? { label: "SETTING UP", tone: "bg-amber-500/15 text-amber-300", dot: "bg-amber-500" }
      : session.state === "returning"
        ? { label: "RETURNING", tone: "bg-sky-500/15 text-sky-300", dot: "bg-sky-500" }
        : current
          ? { label: "RUNNING", tone: "bg-green-500/15 text-green-300", dot: "bg-green-500" }
          : { label: "READY", tone: "bg-slate-500/15 text-slate-200", dot: "bg-slate-300" };
    const commandSeconds = current ? elapsed(current) : 0;
    const commandPct =
      current?.status === "running" && session.command_time_s > 0
        ? Math.min(100, (commandSeconds / session.command_time_s) * 100)
        : 0;
    return (
      <div className="min-h-screen bg-black text-white flex flex-col p-4 sm:p-6 lg:p-8">
        {header}
        <div className="flex-1 flex justify-center">
          <div className="bg-gray-900 rounded-lg border border-gray-700 p-5 sm:p-8 w-full max-w-2xl space-y-5 self-start">
            <div className="flex items-center justify-between gap-4">
              <div
                className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold tracking-widest ${pill.tone}`}
              >
                <span className={`w-2 h-2 rounded-full ${pill.dot} animate-pulse`} />
                {pill.label}
              </div>
              <div className="font-mono text-2xl font-bold text-gray-100" aria-label="Command time">
                {settingUp
                  ? formatTime(setupElapsed)
                  : current?.status === "running"
                    ? `${formatTime(commandSeconds)} / ${formatTime(session.command_time_s)}`
                    : ""}
              </div>
            </div>
            <p className="text-sm text-gray-400">
              {settingUp
                ? `${session.phase ? `${session.phase[0].toUpperCase()}${session.phase.slice(1)}` : "Loading the policy & connecting the arm"}…`
                : session.state === "returning"
                  ? "Going back to the start pose…"
                  : current?.status === "pending"
                    ? `Starting “${current.text}”…`
                    : current
                      ? `Running “${current.text}”`
                      : "Ready. Type a command below and press Enter."}
            </p>
            <div className="w-full bg-gray-800 rounded-full h-1.5">
              <div
                className={`h-1.5 rounded-full transition-all duration-300 ${
                  settingUp ? "bg-amber-500/40 animate-pulse w-full" : "bg-green-500"
                }`}
                style={settingUp ? undefined : { width: `${commandPct}%` }}
              />
            </div>

            <ArmLinkNotice issues={status.arm_link_issues} />

            <CommandHistory
              commands={session.commands}
              commandTimeS={session.command_time_s}
              elapsed={elapsed}
              onEnd={handleEnd}
              ending={ending}
            />
            <CommandInput blockedReason={blockedReason(status, session)} onSend={handleSend} />
            <SessionBlendingToggle
              enabled={blendPending ?? session.chunk_blending}
              blendSteps={session.blend_steps}
              locked={session.state === "running"}
              busy={blendPending !== null}
              onToggle={handleBlending}
            />

            {policyLine}

            <Button
              onClick={() => setShowStopConfirm(true)}
              disabled={!status.inference_active}
              className="w-full bg-red-500 hover:bg-red-600 text-white font-semibold py-5 disabled:opacity-50"
            >
              <Square className="w-4 h-4 mr-2" />
              End session
            </Button>
          </div>
        </div>
        {stopDialog}
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black text-white flex flex-col p-4 sm:p-6 lg:p-8">
      {header}

      <div className="flex-1 flex items-center justify-center">
        <div className="bg-gray-900 rounded-lg border border-gray-700 p-8 w-full max-w-xl">
          <div className="text-center mb-6">
            <div
              className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold tracking-widest ${
                isSettingUp
                  ? "bg-amber-500/15 text-amber-300"
                  : "bg-green-500/15 text-green-300"
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  isSettingUp ? "bg-amber-500" : "bg-green-500"
                } animate-pulse`}
              />
              {pillLabel}
            </div>
          </div>

          <div className="text-center mb-4">
            <div
              className={`text-7xl font-mono font-bold leading-none ${
                isSettingUp ? "text-amber-400" : "text-green-400"
              }`}
            >
              {formatTime(timerSeconds)}
            </div>
            <div className="text-sm text-gray-500 mt-2">
              {isSettingUp
                ? "Loading policy & connecting hardware…"
                : `/ ${formatTime(duration)}`}
            </div>
          </div>

          <div className="w-full bg-gray-800 rounded-full h-1.5 mb-8">
            <div
              className={`h-1.5 rounded-full transition-all duration-500 ${
                isSettingUp
                  ? "bg-amber-500/40 animate-pulse w-full"
                  : "bg-green-500"
              }`}
              style={isSettingUp ? undefined : { width: `${pct}%` }}
            />
          </div>

          <ArmLinkNotice issues={status.arm_link_issues} className="mb-4" />

          <div className="mb-6">{policyLine}</div>

          <Button
            onClick={() => setShowStopConfirm(true)}
            disabled={!status.inference_active}
            className="w-full bg-red-500 hover:bg-red-600 text-white font-semibold py-6 text-lg disabled:opacity-50"
          >
            <Square className="w-5 h-5 mr-2" />
            Stop
          </Button>
        </div>
      </div>

      {stopDialog}
    </div>
  );
};

export default Inference;
