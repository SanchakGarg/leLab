import React from "react";
import { AlertTriangle } from "lucide-react";

/** An arm whose USB link is failing now, or failed for a while recently (from the backend's bus guard). */
export interface ArmLinkIssue {
  device: string;
  ongoing: boolean;
  duration_s: number;
  skipped?: number;
}

interface Props {
  issues?: ArmLinkIssue[] | null;
  /** While recording, a past gap means repeated frames in the episode, so suggest re-recording. */
  recording?: boolean;
  className?: string;
}

const seconds = (s: number) => `${s < 10 ? s.toFixed(1) : Math.round(s)} s`;

/** A small amber notice: LeLab keeps going through brief USB glitches, and stops after 10 s without an answer. */
const ArmLinkNotice: React.FC<Props> = ({ issues, recording = false, className = "" }) => {
  if (!issues?.length) return null;
  return (
    <div
      role="status"
      className={`flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-200 ${className}`}
    >
      <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0" />
      <div className="space-y-0.5">
        {issues.map((issue) => (
          <p key={issue.device}>
            {issue.ongoing
              ? `The ${issue.device} isn't answering (${seconds(issue.duration_s)}). Check its USB cable; LeLab stops after 10 s.`
              : recording
                ? `The ${issue.device} stopped answering for ${seconds(issue.duration_s)} in this phase; those frames repeat its last position. Re-record the episode if that matters.`
                : `The ${issue.device} stopped answering for ${seconds(issue.duration_s)}.`}
          </p>
        ))}
      </div>
    </div>
  );
};

export default ArmLinkNotice;
