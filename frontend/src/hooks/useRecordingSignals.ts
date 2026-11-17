import { useEffect, useRef, useState } from "react";
import { useApi } from "@/contexts/ApiContext";

/** One frame of joint positions, keyed by joint name (`shoulder_pan`, ...). */
export interface SignalSample {
  t: number; // seconds since the epoch, from the server
  joints: Record<string, number>;
}

export interface RecordingSignals {
  action: SignalSample[];
  observation: SignalSample[];
}

interface SignalsResponse extends RecordingSignals {
  seq: number;
}

const POLL_MS = 200;
const EMPTY: RecordingSignals = { action: [], observation: [] };

/** Keep the samples of the last `windowS` seconds, measured from the newest. */
export function trimWindow(samples: SignalSample[], windowS: number): SignalSample[] {
  if (samples.length === 0) return samples;
  const cutoff = samples[samples.length - 1].t - windowS;
  const first = samples.findIndex((s) => s.t >= cutoff);
  return first <= 0 ? samples : samples.slice(first);
}

/**
 * The action sent to the follower and the observation read from it during a
 * recording session, for the live graphs. Polls `/recording-signals` for the
 * samples newer than the last one seen while `active`, keeping a rolling
 * window of `windowS` seconds.
 */
export function useRecordingSignals(active: boolean, windowS = 10): RecordingSignals {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [signals, setSignals] = useState<RecordingSignals>(EMPTY);
  const seqRef = useRef(0);

  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | null = null;

    const poll = async () => {
      try {
        const r = await fetchWithHeaders(`${baseUrl}/recording-signals?since=${seqRef.current}`, {
          signal: controller.signal,
        });
        if (r.ok) {
          const data = (await r.json()) as SignalsResponse;
          if (controller.signal.aborted) return;
          seqRef.current = data.seq;
          if (data.action.length > 0 || data.observation.length > 0) {
            setSignals((prev) => ({
              action: trimWindow([...prev.action, ...data.action], windowS),
              observation: trimWindow([...prev.observation, ...data.observation], windowS),
            }));
          }
        }
      } catch {
        // Aborted, or the server is briefly unreachable: try again next tick.
      }
      if (!controller.signal.aborted) timer = setTimeout(poll, POLL_MS);
    };
    void poll();

    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [active, baseUrl, fetchWithHeaders, windowS]);

  return signals;
}
