import React, { useMemo, useState } from "react";
import { CartesianGrid, Line, LineChart, Tooltip, XAxis, YAxis } from "recharts";
import { ChevronDown, ChevronRight, LineChart as LineChartIcon, X } from "lucide-react";
import type { RecordingSignals, SignalSample } from "@/hooks/useRecordingSignals";
import { JOINTS, toRows } from "@/lib/signalGraph";

const HIDDEN_KEY = "lelab.recordingGraphsHidden";
const CHART_WIDTH = 248;
const CHART_HEIGHT = 92;

const readHidden = (): boolean => {
  try {
    return window.localStorage.getItem(HIDDEN_KEY) === "1";
  } catch {
    return false;
  }
};

const writeHidden = (hidden: boolean) => {
  try {
    window.localStorage.setItem(HIDDEN_KEY, hidden ? "1" : "0");
  } catch {
    // Storage unavailable: the choice lasts for this page only.
  }
};

interface WindowProps {
  title: string;
  description: string;
  samples: SignalSample[];
  now: number;
}

const SignalWindow: React.FC<WindowProps> = ({ title, description, samples, now }) => {
  const [collapsed, setCollapsed] = useState(false);
  const rows = useMemo(() => toRows(samples, now), [samples, now]);
  const Chevron = collapsed ? ChevronRight : ChevronDown;

  return (
    <section
      aria-label={`${title} graph`}
      className="rounded-md border border-gray-800 bg-gray-950/95 px-1 pb-1 pt-1"
    >
      <button
        type="button"
        onClick={() => setCollapsed((c) => !c)}
        aria-expanded={!collapsed}
        title={description}
        className="flex w-full items-center gap-1 px-1.5 pb-0.5 text-left text-[10px] font-semibold uppercase tracking-wider text-gray-400 hover:text-gray-200"
      >
        <Chevron className="h-3 w-3" />
        {title}
      </button>
      {!collapsed &&
        (rows.length === 0 ? (
          <div
            className="flex items-center justify-center text-[10px] text-gray-600"
            style={{ width: CHART_WIDTH, height: CHART_HEIGHT }}
          >
            Waiting for the arms…
          </div>
        ) : (
          <LineChart
            width={CHART_WIDTH}
            height={CHART_HEIGHT}
            data={rows}
            margin={{ top: 2, right: 4, bottom: 0, left: 0 }}
          >
            <CartesianGrid stroke="#1f2937" strokeDasharray="2 3" vertical={false} />
            <XAxis dataKey="t" type="number" domain={["dataMin", 0]} hide />
            <YAxis
              width={30}
              tickCount={3}
              domain={["auto", "auto"]}
              tick={{ fontSize: 9, fill: "#6b7280" }}
              tickFormatter={(v: number) => v.toFixed(0)}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                background: "#0a0a0a",
                border: "1px solid #1f2937",
                borderRadius: 6,
                fontSize: 10,
                padding: "2px 6px",
              }}
              labelStyle={{ color: "#9ca3af" }}
              labelFormatter={(t: number) => `${t.toFixed(1)} s`}
              formatter={(v: number, name: string) => [v.toFixed(1), name]}
            />
            {JOINTS.map((j) => (
              <Line
                key={j.key}
                type="monotone"
                dataKey={j.key}
                name={j.label}
                stroke={j.color}
                strokeWidth={1.25}
                dot={false}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        ))}
    </section>
  );
};

interface Props {
  signals: RecordingSignals;
}

/**
 * Two small live graphs for the recording page: the action sent to the
 * follower (the leader's command) and the observation read from it, all six
 * joints over the last few seconds. Hiding them is remembered.
 */
const SignalGraphs: React.FC<Props> = ({ signals }) => {
  const [hidden, setHidden] = useState(readHidden);
  const toggle = (next: boolean) => {
    setHidden(next);
    writeHidden(next);
  };

  // One clock for both windows, so their time axes line up.
  const now = Math.max(signals.action.at(-1)?.t ?? 0, signals.observation.at(-1)?.t ?? 0);

  if (hidden) {
    return (
      <button
        type="button"
        onClick={() => toggle(false)}
        className="flex flex-shrink-0 items-center gap-1.5 self-end rounded-md border border-gray-800 bg-gray-950 px-2 py-1 text-[11px] text-gray-400 hover:text-white"
      >
        <LineChartIcon className="h-3.5 w-3.5" />
        Show graphs
      </button>
    );
  }

  return (
    <div className="flex flex-shrink-0 flex-col gap-1.5 self-end" style={{ width: CHART_WIDTH + 10 }}>
      <div className="flex items-center justify-between gap-2 px-0.5">
        <ul className="flex flex-wrap gap-x-2 gap-y-0.5 text-[10px] text-gray-500" aria-label="Joint colours">
          {JOINTS.map((j) => (
            <li key={j.key} className="flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: j.color }} />
              {j.label}
            </li>
          ))}
        </ul>
        <button
          type="button"
          onClick={() => toggle(true)}
          aria-label="Hide graphs"
          title="Hide graphs"
          className="rounded p-0.5 text-gray-500 hover:bg-gray-800 hover:text-white"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      <SignalWindow
        title="Action"
        description="Joint positions sent to the follower (the leader arm's command)"
        samples={signals.action}
        now={now}
      />
      <SignalWindow
        title="Observation"
        description="Joint positions read from the follower arm"
        samples={signals.observation}
        now={now}
      />
    </div>
  );
};

export default SignalGraphs;
