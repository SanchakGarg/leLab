import React from "react";
import { MessageSquareText } from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { NumberInput } from "@/components/ui/number-input";
import { COMMAND_TIME_MAX_S, COMMAND_TIME_MIN_S, type RunSettings } from "@/hooks/useRunSettings";

interface Props {
  idPrefix: string;
  settings: Pick<RunSettings, "interactive" | "commandTimeS" | "returnBetweenCommands" | "autoComplete">;
  onChange: (patch: Partial<RunSettings>) => void;
  /** The policy type, when it doesn't read text (ACT and the like): every command runs the same skill. */
  ignoresTextType?: string | null;
}

const checkboxClass =
  "mt-0.5 border-gray-500 data-[state=checked]:bg-emerald-500 data-[state=checked]:border-emerald-500";

/** "Interactive session" and, when it is on, the time per command and how commands end. */
const InteractiveOptions: React.FC<Props> = ({ idPrefix, settings, onChange, ignoresTextType }) => (
  <div className="space-y-3 rounded-md border border-gray-800 p-3">
    <div className="flex items-start gap-3">
      <Checkbox
        id={`${idPrefix}-interactive`}
        checked={settings.interactive}
        onCheckedChange={(value) => onChange({ interactive: value === true })}
        className={checkboxClass}
      />
      <div className="space-y-1">
        <Label
          htmlFor={`${idPrefix}-interactive`}
          className="flex items-center gap-1.5 text-sm font-medium text-gray-200 cursor-pointer"
        >
          <MessageSquareText className="h-4 w-4" />
          Interactive session
        </Label>
        <p className="text-xs text-gray-500">
          Type commands one at a time while the arm waits. Each one runs until it's done or its time is up.
        </p>
      </div>
    </div>
    {settings.interactive ? (
      <div className="space-y-3 sm:pl-7">
        <div className="space-y-1.5">
          <Label htmlFor={`${idPrefix}-command-time`} className="text-sm text-gray-300">
            Time per command (seconds)
          </Label>
          <NumberInput
            id={`${idPrefix}-command-time`}
            min={COMMAND_TIME_MIN_S}
            max={COMMAND_TIME_MAX_S}
            value={settings.commandTimeS}
            onChange={(v) => {
              if (v !== undefined) onChange({ commandTimeS: v });
            }}
            className="bg-gray-800 border-gray-700 text-white"
          />
          <p className="text-xs text-gray-500">
            A command not done by then counts as didn't complete ({COMMAND_TIME_MIN_S}–{COMMAND_TIME_MAX_S} s).
          </p>
        </div>
        <div className="flex items-start gap-3">
          <Checkbox
            id={`${idPrefix}-return-between`}
            checked={settings.returnBetweenCommands}
            onCheckedChange={(value) => onChange({ returnBetweenCommands: value === true })}
            className={checkboxClass}
          />
          <div className="space-y-1">
            <Label htmlFor={`${idPrefix}-return-between`} className="text-sm text-gray-200 cursor-pointer">
              Return to the start pose after each command
            </Label>
            <p className="text-xs text-gray-500">
              Off: the next command starts where the last one ended, so it can build on it.
            </p>
          </div>
        </div>
        <div className="flex items-start gap-3">
          <Checkbox
            id={`${idPrefix}-auto-complete`}
            checked={settings.autoComplete}
            onCheckedChange={(value) => onChange({ autoComplete: value === true })}
            className={checkboxClass}
          />
          <div className="space-y-1">
            <Label htmlFor={`${idPrefix}-auto-complete`} className="text-sm text-gray-200 cursor-pointer">
              Count a command as done when the arm stops moving
            </Label>
            <p className="text-xs text-gray-500">
              The model can't tell when a task is finished, so this is a guess. You can always press Done.
            </p>
          </div>
        </div>
        {ignoresTextType ? (
          <p className="text-xs text-amber-300/90">
            This model ({ignoresTextType}) doesn't read text: every command runs the same skill.
          </p>
        ) : null}
      </div>
    ) : null}
  </div>
);

export default InteractiveOptions;
