import React, { useLayoutEffect, useRef } from "react";
import { AlertTriangle, CheckCircle2, CircleSlash, Loader2, Square, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { CommandStatus, InteractiveCommand } from "@/lib/inferenceApi";
import { cn } from "@/lib/utils";

function clock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

const LABELS: Record<CommandStatus, string> = {
  pending: "Starting",
  running: "Running",
  completed: "Done",
  not_completed: "Didn't complete",
  stopped: "Stopped",
  failed: "Error",
};

const TONES: Record<CommandStatus, string> = {
  pending: "text-amber-300",
  running: "text-amber-300",
  completed: "text-green-400",
  not_completed: "text-red-400",
  stopped: "text-slate-400",
  failed: "text-red-400",
};

function StatusIcon({ status }: { status: CommandStatus }) {
  const className = cn("h-4 w-4 shrink-0", TONES[status]);
  switch (status) {
    case "pending":
    case "running":
      return <Loader2 className={cn(className, "animate-spin")} aria-hidden />;
    case "completed":
      return <CheckCircle2 className={className} aria-hidden />;
    case "not_completed":
      return <XCircle className={className} aria-hidden />;
    case "stopped":
      return <CircleSlash className={className} aria-hidden />;
    default:
      return <AlertTriangle className={className} aria-hidden />;
  }
}

/** Why a finished command ended, in words. */
function detail(command: InteractiveCommand, commandTimeS: number): string | null {
  switch (command.how) {
    case null:
      return command.status === "pending" ? "waiting for the model" : null;
    case "arm settled":
      return "the arm stopped moving";
    case "marked done":
      return "marked done";
    case "time limit":
      return `not done within ${clock(commandTimeS)}`;
    case "stopped":
      return "stopped by you";
    default:
      return command.how;
  }
}

interface Props {
  commands: InteractiveCommand[];
  commandTimeS: number;
  /** Elapsed seconds of a running command, ticking between status polls. */
  elapsed?: (command: InteractiveCommand) => number;
  /** Done / Stop command for the running command; omitted for a finished session. */
  onEnd?: (completed: boolean) => void;
  ending?: boolean;
  className?: string;
}

// Within this many pixels of the bottom counts as "at the bottom": new commands then scroll into view.
const STICKY_PX = 32;

/**
 * Every command of an interactive session, oldest first. It follows the newest command while you are at
 * the bottom of the list, and stays put when you have scrolled up to read older ones.
 */
const CommandHistory: React.FC<Props> = ({ commands, commandTimeS, elapsed, onEnd, ending, className }) => {
  const listRef = useRef<HTMLDivElement>(null);
  const stickRef = useRef(true);

  // After every render: a new command, or a row that grew (its detail line, the buttons), stays in view
  // while following the bottom.
  useLayoutEffect(() => {
    const list = listRef.current;
    if (list && stickRef.current) list.scrollTop = list.scrollHeight;
  });

  const onScroll = () => {
    const list = listRef.current;
    if (list) stickRef.current = list.scrollHeight - list.scrollTop - list.clientHeight < STICKY_PX;
  };

  if (commands.length === 0) {
    return (
      <div className={cn("rounded-md border border-gray-800 px-4 py-6 text-center text-sm text-gray-500", className)}>
        No commands yet. Type one below and press Enter.
      </div>
    );
  }

  return (
    <div
      ref={listRef}
      onScroll={onScroll}
      data-testid="command-history"
      className={cn("max-h-80 overflow-y-auto rounded-md border border-gray-800 divide-y divide-gray-800", className)}
    >
      {commands.map((command) => {
        const active = command.status === "pending" || command.status === "running";
        const seconds = active && elapsed ? elapsed(command) : command.elapsed_s;
        const why = detail(command, commandTimeS);
        return (
          <div
            key={command.id}
            data-testid="command-row"
            data-status={command.status}
            className={cn("px-3 py-2.5", active && "bg-amber-500/5")}
          >
            <div className="flex items-start gap-3">
              <span className="w-6 pt-0.5 text-right font-mono text-xs text-gray-600">{command.id}</span>
              <StatusIcon status={command.status} />
              <div className="min-w-0 flex-1">
                <p className="break-words text-sm text-gray-100">{command.text}</p>
                <p className="mt-0.5 text-xs">
                  <span className={cn("font-medium", TONES[command.status])}>{LABELS[command.status]}</span>
                  {why ? <span className="text-gray-500"> · {why}</span> : null}
                </p>
              </div>
              <span className="shrink-0 pt-0.5 font-mono text-xs text-gray-400">
                {command.status === "running"
                  ? `${clock(seconds)} / ${clock(commandTimeS)}`
                  : command.started_at != null
                    ? clock(seconds)
                    : ""}
              </span>
            </div>
            {active && onEnd ? (
              <div className="mt-2 flex justify-end gap-2">
                {command.status === "running" ? (
                  <Button
                    size="sm"
                    disabled={ending}
                    onClick={() => onEnd(true)}
                    className="h-8 bg-green-600 text-white hover:bg-green-500"
                  >
                    <CheckCircle2 className="mr-1.5 h-4 w-4" />
                    Done
                  </Button>
                ) : null}
                <Button
                  size="sm"
                  variant="outline"
                  disabled={ending}
                  onClick={() => onEnd(false)}
                  className="h-8 border-gray-600 bg-gray-950 text-gray-200 hover:bg-gray-800"
                >
                  <Square className="mr-1.5 h-3.5 w-3.5" />
                  Stop command
                </Button>
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
};

export default CommandHistory;
