import React, { createContext, ReactNode, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { useApi } from "./ApiContext";
import { RemoteServer, listRemoteServers } from "@/lib/remoteApi";

const SELECTED_KEY = "lelab.remoteServer";

interface RemoteServerValue {
  /** Saved servers, each with its live SSH session state. */
  servers: RemoteServer[];
  loading: boolean;
  refresh: () => Promise<void>;
  /** The server the UI uses by default (last picked, else the first saved). */
  selected: RemoteServer | null;
  selectServer: (name: string | null) => void;
}

const RemoteServerContext = createContext<RemoteServerValue | undefined>(undefined);

const readSelected = (): string | null => {
  try {
    return window.localStorage.getItem(SELECTED_KEY);
  } catch {
    return null;
  }
};

export const RemoteServerProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [servers, setServers] = useState<RemoteServer[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedName, setSelectedName] = useState<string | null>(readSelected);

  const refresh = useCallback(async () => {
    try {
      setServers(await listRemoteServers(baseUrl, fetchWithHeaders));
    } catch (err) {
      // An older backend without these routes, or none reachable: no servers.
      console.warn("Remote servers fetch failed:", err);
      setServers([]);
    } finally {
      setLoading(false);
    }
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const selectServer = useCallback((name: string | null) => {
    setSelectedName(name);
    try {
      if (name) window.localStorage.setItem(SELECTED_KEY, name);
      else window.localStorage.removeItem(SELECTED_KEY);
    } catch {
      // storage unavailable: the choice lasts for this page only
    }
  }, []);

  const value = useMemo(() => {
    const selected = servers.find((s) => s.name === selectedName) ?? servers[0] ?? null;
    return { servers, loading, refresh, selected, selectServer };
  }, [servers, loading, refresh, selectedName, selectServer]);

  return <RemoteServerContext.Provider value={value}>{children}</RemoteServerContext.Provider>;
};

export const useRemoteServers = (): RemoteServerValue => {
  const context = useContext(RemoteServerContext);
  if (context === undefined) {
    throw new Error("useRemoteServers must be used within a RemoteServerProvider");
  }
  return context;
};

/** The remote servers, or null outside a RemoteServerProvider (for components also used without one). */
export const useOptionalRemoteServers = (): RemoteServerValue | null => useContext(RemoteServerContext) ?? null;
