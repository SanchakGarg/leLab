import React from "react";
import { Blend } from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { NumberInput } from "@/components/ui/number-input";
import { BLEND_STEPS_MAX, BLEND_STEPS_MIN, type RunSettings } from "@/hooks/useRunSettings";

const checkboxClass =
  "mt-0.5 border-gray-500 data-[state=checked]:bg-emerald-500 data-[state=checked]:border-emerald-500";

const seconds = (actions: number, fps: number) => `${(actions / fps).toFixed(actions / fps < 1 ? 2 : 1)} s`;

interface Props {
  id: string;
  settings: Pick<RunSettings, "chunkBlending" | "blendSteps">;
  onChange: (patch: Partial<RunSettings>) => void;
  /** The control rate, to say how long the blend lasts. */
  fps?: number;
  /** A line about what turning it on changes (a local run goes through LeLab's own loop). */
  note?: string | null;
  /** Why it would have no effect with the other settings. */
  warning?: string | null;
}

/** "Blend chunks" in a start dialog: on/off, and over how many actions. */
const ChunkBlendingOption: React.FC<Props> = ({ id, settings, onChange, fps = 30, note, warning }) => (
  <div className="space-y-3 rounded-md border border-gray-800 p-3">
    <div className="flex items-start gap-3">
      <Checkbox
        id={id}
        checked={settings.chunkBlending}
        onCheckedChange={(value) => onChange({ chunkBlending: value === true })}
        className={checkboxClass}
      />
      <div className="space-y-1">
        <Label htmlFor={id} className="flex items-center gap-1.5 text-sm font-medium text-gray-200 cursor-pointer">
          <Blend className="h-4 w-4" />
          Blend chunks
        </Label>
        <p className="text-xs text-gray-500">
          The model plans actions in chunks. When a new chunk starts away from where the last one was heading, the
          arm jerks. Blending fades from the old chunk into the new one instead.
        </p>
      </div>
    </div>
    {settings.chunkBlending ? (
      <div className="space-y-1.5 sm:pl-7">
        <Label htmlFor={`${id}-steps`} className="text-sm text-gray-300">
          Blend over (actions)
        </Label>
        <NumberInput
          id={`${id}-steps`}
          min={BLEND_STEPS_MIN}
          max={BLEND_STEPS_MAX}
          value={settings.blendSteps}
          onChange={(v) => {
            if (v !== undefined) onChange({ blendSteps: v });
          }}
          className="bg-gray-800 border-gray-700 text-white"
        />
        <p className="text-xs text-gray-500">
          About {seconds(settings.blendSteps, fps)} at {fps} Hz. Longer is smoother but follows the new chunk later.
        </p>
        {note ? <p className="text-xs text-gray-400">{note}</p> : null}
        {warning ? <p className="text-xs text-amber-300/90">{warning}</p> : null}
      </div>
    ) : null}
  </div>
);

interface SessionProps {
  enabled: boolean;
  blendSteps: number;
  /** A command is running: blending changes only between commands. */
  locked: boolean;
  busy: boolean;
  onToggle: (enabled: boolean) => void;
}

/** Chunk blending on the session page, switchable between commands. */
export const SessionBlendingToggle: React.FC<SessionProps> = ({ enabled, blendSteps, locked, busy, onToggle }) => (
  <div className="flex items-start gap-3 rounded-md border border-gray-800 px-3 py-2.5">
    <Checkbox
      id="session-chunk-blending"
      checked={enabled}
      disabled={locked || busy}
      onCheckedChange={(value) => onToggle(value === true)}
      className={checkboxClass}
    />
    <div className="space-y-0.5">
      <Label
        htmlFor="session-chunk-blending"
        className={`flex items-center gap-1.5 text-sm font-medium ${locked ? "text-gray-500" : "text-gray-200 cursor-pointer"}`}
      >
        <Blend className="h-4 w-4" />
        Blend chunks
      </Label>
      <p className="text-xs text-gray-500">
        {locked
          ? "Change it between commands."
          : enabled
            ? `Fades into each new chunk over ${blendSteps} actions. Applies from the next command.`
            : "Each new chunk takes over at once. Applies from the next command."}
      </p>
    </div>
  </div>
);

export default ChunkBlendingOption;
