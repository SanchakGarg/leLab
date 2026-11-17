import React, { useState } from "react";
import { CheckCircle, KeyRound, Loader2, ShieldAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useApi } from "@/contexts/ApiContext";
import { useRemoteServers } from "@/contexts/RemoteServerContext";
import {
  HostKeyChallenge,
  RemoteServer,
  connectRemoteServer,
  disconnectRemoteServer,
  errorText,
  hostKeyChallenge,
} from "@/lib/remoteApi";

interface Props {
  server: RemoteServer;
  /** Offer Disconnect while connected (the settings dialog does; inline uses don't). */
  allowDisconnect?: boolean;
}

/**
 * Connect to a saved server: asks for the password (or key passphrase), and on a
 * first connection shows the server's host key fingerprint for the user to confirm.
 * The password only lives in this form until it is sent.
 */
const ServerConnectForm: React.FC<Props> = ({ server, allowDisconnect = false }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { refresh, selectServer } = useRemoteServers();
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [challenge, setChallenge] = useState<HostKeyChallenge | null>(null);

  const connect = async (trustFingerprint?: string) => {
    setBusy(true);
    setError(null);
    try {
      await connectRemoteServer(baseUrl, fetchWithHeaders, server.name, {
        ...(password ? { password } : {}),
        ...(trustFingerprint ? { trust_fingerprint: trustFingerprint } : {}),
      });
      setPassword("");
      setChallenge(null);
      selectServer(server.name); // the last server connected to is the default next time
      await refresh();
    } catch (e) {
      const hostKey = hostKeyChallenge(e);
      if (hostKey) setChallenge(hostKey);
      else setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const disconnect = async () => {
    setBusy(true);
    setError(null);
    try {
      await disconnectRemoteServer(baseUrl, fetchWithHeaders, server.name);
      await refresh();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  if (server.session.connected) {
    return (
      <div className="space-y-2">
        <div className="flex items-center justify-between gap-3 rounded-md border border-emerald-800/60 bg-emerald-950/30 px-3 py-2 text-sm">
          <span className="flex items-center gap-2 text-emerald-200">
            <CheckCircle className="h-4 w-4" />
            Connected to {server.username}@{server.host}
            {server.session.home ? <span className="text-emerald-400/70">({server.session.home})</span> : null}
          </span>
          {allowDisconnect ? (
            <Button
              size="sm"
              variant="outline"
              onClick={disconnect}
              disabled={busy || server.session.in_use}
              className="h-7 border-gray-700 bg-gray-900 text-xs text-gray-300 hover:bg-gray-800"
            >
              Disconnect
            </Button>
          ) : null}
        </div>
        {error ? <p className="text-xs text-red-300">{error}</p> : null}
      </div>
    );
  }

  const needsSecret = server.auth !== "agent";
  return (
    <div className="space-y-3">
      {needsSecret ? (
        <div className="space-y-1.5">
          <Label htmlFor={`pw-${server.name}`} className="text-sm text-gray-300">
            {server.auth === "password" ? `Password for ${server.username}@${server.host}` : "Key passphrase (if it has one)"}
          </Label>
          <Input
            id={`pw-${server.name}`}
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !busy) connect();
            }}
            className="bg-gray-800 border-gray-700 text-white"
          />
          <p className="text-xs text-gray-500">Kept in memory while LeLab runs, never saved to disk.</p>
        </div>
      ) : null}

      {challenge ? (
        <div className="space-y-2 rounded-md border border-amber-700/60 bg-amber-950/30 p-3 text-sm text-amber-100">
          <p className="flex items-center gap-2 font-medium">
            <ShieldAlert className="h-4 w-4" />
            First connection to {server.host}
          </p>
          <p className="text-xs text-amber-200/80">
            Its {challenge.keyType} host key fingerprint is:
          </p>
          <code className="block break-all rounded bg-black/40 px-2 py-1 text-xs">{challenge.fingerprint}</code>
          <p className="text-xs text-amber-200/80">
            Trust it only if it is your server's (ask its administrator, or compare with{" "}
            <code>ssh-keygen -lf /etc/ssh/ssh_host_*_key.pub</code> on the server).
          </p>
          <Button
            size="sm"
            onClick={() => connect(challenge.fingerprint)}
            disabled={busy}
            className="bg-amber-600 text-white hover:bg-amber-500"
          >
            Trust and connect
          </Button>
        </div>
      ) : (
        <Button
          onClick={() => connect()}
          disabled={busy || (server.auth === "password" && !password)}
          className="bg-emerald-600 text-white hover:bg-emerald-500"
        >
          {busy ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <KeyRound className="mr-2 h-4 w-4" />}
          Connect
        </Button>
      )}
      {error ? <p className="text-xs text-red-300">{error}</p> : null}
    </div>
  );
};

export default ServerConnectForm;
