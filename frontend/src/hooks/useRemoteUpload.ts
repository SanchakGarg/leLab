import { useCallback, useEffect, useRef, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import {
  RemoteUploadRequest,
  RemoteUploadStatus,
  cancelRemoteUpload,
  errorText,
  getRemoteUploadStatus,
  startRemoteUpload,
} from "@/lib/remoteApi";

const POLL_MS = 1000;

/**
 * Drives one SFTP upload: starts it, polls its progress while it runs, and
 * picks up an upload that was already running when the page opened.
 */
export function useRemoteUpload(enabled: boolean = true) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [status, setStatus] = useState<RemoteUploadStatus>({ state: "idle" });
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const poll = useCallback(async () => {
    try {
      const next = await getRemoteUploadStatus(baseUrl, fetchWithHeaders);
      setStatus(next);
      if (next.state === "running") timer.current = setTimeout(poll, POLL_MS);
    } catch (e) {
      setError(errorText(e));
    }
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    if (!enabled) return;
    poll();
    return () => {
      if (timer.current) clearTimeout(timer.current);
      timer.current = null;
    };
  }, [enabled, poll]);

  const start = useCallback(
    async (request: RemoteUploadRequest) => {
      setStarting(true);
      setError(null);
      try {
        const next = await startRemoteUpload(baseUrl, fetchWithHeaders, request);
        setStatus(next);
        if (timer.current) clearTimeout(timer.current);
        timer.current = setTimeout(poll, POLL_MS);
        return true;
      } catch (e) {
        setError(errorText(e));
        return false;
      } finally {
        setStarting(false);
      }
    },
    [baseUrl, fetchWithHeaders, poll],
  );

  const cancel = useCallback(async () => {
    try {
      await cancelRemoteUpload(baseUrl, fetchWithHeaders);
    } catch (e) {
      setError(errorText(e));
    }
  }, [baseUrl, fetchWithHeaders]);

  return { status, error, starting, start, cancel };
}
