import React, { useEffect, useState } from "react";
import { Copy, Loader2, Server, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useApi } from "@/contexts/ApiContext";
import { useRemoteServers } from "@/contexts/RemoteServerContext";
import {
  RemoteAuth,
  RemoteServerProfile,
  deleteRemoteServer,
  emptyProfile,
  errorText,
  installPolicyServer,
  saveRemoteServer,
} from "@/lib/remoteApi";
import ServerConnectForm from "./ServerConnectForm";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const NEW = "__new__";

/** Add, edit, connect to and remove the servers datasets are sent to and models are served from. */
const RemoteServerDialog: React.FC<Props> = ({ open, onOpenChange }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { servers, selected, selectServer, refresh } = useRemoteServers();
  const [editing, setEditing] = useState<string>(NEW);
  const [name, setName] = useState("");
  const [profile, setProfile] = useState<RemoteServerProfile>(emptyProfile);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [install, setInstall] = useState<{ busy: boolean; result?: string; error?: string }>({ busy: false });

  // Open on the selected server, or on a blank form when none is saved.
  useEffect(() => {
    if (!open) return;
    setEditing(selected?.name ?? NEW);
    setError(null);
    setInstall({ busy: false });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => {
    const server = servers.find((s) => s.name === editing);
    if (server) {
      const { name: serverName, session: _session, ...rest } = server;
      setName(serverName);
      setProfile(rest);
    } else {
      setName(servers.length === 0 ? "cluster" : "");
      setProfile(emptyProfile());
    }
  }, [editing, servers]);

  const current = servers.find((s) => s.name === editing) ?? null;
  const set = <K extends keyof RemoteServerProfile>(key: K, value: RemoteServerProfile[K]) =>
    setProfile((p) => ({ ...p, [key]: value }));

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      await saveRemoteServer(baseUrl, fetchWithHeaders, name.trim(), profile);
      await refresh();
      selectServer(name.trim());
      setEditing(name.trim());
    } catch (e) {
      setError(errorText(e));
    } finally {
      setSaving(false);
    }
  };

  const remove = async () => {
    if (!current) return;
    setSaving(true);
    setError(null);
    try {
      await deleteRemoteServer(baseUrl, fetchWithHeaders, current.name);
      await refresh();
      selectServer(null);
      setEditing(NEW);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setSaving(false);
    }
  };

  const copyServerScript = async () => {
    if (!current) return;
    setInstall({ busy: true });
    try {
      const result = await installPolicyServer(baseUrl, fetchWithHeaders, current.name, profile.remote_work_dir);
      setInstall({ busy: false, result: result.command });
    } catch (e) {
      setInstall({ busy: false, error: errorText(e) });
    }
  };

  const dirty =
    !current ||
    (Object.keys(profile) as (keyof RemoteServerProfile)[]).some((k) => profile[k] !== current[k]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-gray-900 border-gray-800 text-white sm:max-w-[560px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-white">
            <Server className="h-5 w-5" />
            Remote server
          </DialogTitle>
          <DialogDescription className="text-gray-400">
            Your own GPU server or cluster: LeLab sends recorded datasets there over SFTP, and runs the
            models you train there on this arm.
          </DialogDescription>
        </DialogHeader>

        {servers.length > 0 ? (
          <div className="space-y-1.5">
            <Label className="text-sm text-gray-300">Server</Label>
            <Select value={editing} onValueChange={setEditing}>
              <SelectTrigger className="bg-gray-800 border-gray-700 text-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-gray-900 border-gray-700 text-white">
                {servers.map((s) => (
                  <SelectItem key={s.name} value={s.name}>
                    {s.name} — {s.username}@{s.host}
                  </SelectItem>
                ))}
                <SelectItem value={NEW}>+ Add a server</SelectItem>
              </SelectContent>
            </Select>
          </div>
        ) : null}

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="space-y-1.5 sm:col-span-2">
            <Label htmlFor="rs-name" className="text-sm text-gray-300">Name</Label>
            <Input
              id="rs-name"
              value={name}
              disabled={!!current}
              onChange={(e) => setName(e.target.value)}
              placeholder="cluster"
              className="bg-gray-800 border-gray-700 text-white"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="rs-host" className="text-sm text-gray-300">IP address or host</Label>
            <Input
              id="rs-host"
              value={profile.host}
              onChange={(e) => set("host", e.target.value)}
              placeholder="10.2.100.14"
              className="bg-gray-800 border-gray-700 text-white"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="rs-user" className="text-sm text-gray-300">Username</Label>
            <Input
              id="rs-user"
              value={profile.username}
              onChange={(e) => set("username", e.target.value)}
              autoComplete="username"
              className="bg-gray-800 border-gray-700 text-white"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="rs-port" className="text-sm text-gray-300">SSH port</Label>
            <Input
              id="rs-port"
              inputMode="numeric"
              value={String(profile.port)}
              onChange={(e) => set("port", Number(e.target.value.replace(/\D/g, "")) || 0)}
              className="bg-gray-800 border-gray-700 text-white"
            />
          </div>
          <div className="space-y-1.5">
            <Label className="text-sm text-gray-300">Log in with</Label>
            <Select value={profile.auth} onValueChange={(v) => set("auth", v as RemoteAuth)}>
              <SelectTrigger className="bg-gray-800 border-gray-700 text-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-gray-900 border-gray-700 text-white">
                <SelectItem value="password">Password</SelectItem>
                <SelectItem value="key">Key file</SelectItem>
                <SelectItem value="agent">ssh-agent / default keys</SelectItem>
              </SelectContent>
            </Select>
          </div>
          {profile.auth === "key" ? (
            <div className="space-y-1.5 sm:col-span-2">
              <Label htmlFor="rs-key" className="text-sm text-gray-300">Private key file</Label>
              <Input
                id="rs-key"
                value={profile.key_path}
                onChange={(e) => set("key_path", e.target.value)}
                placeholder="~/.ssh/id_ed25519"
                className="bg-gray-800 border-gray-700 text-white"
              />
            </div>
          ) : null}
          <div className="space-y-1.5 sm:col-span-2">
            <Label htmlFor="rs-data" className="text-sm text-gray-300">Dataset folder on the server</Label>
            <Input
              id="rs-data"
              value={profile.remote_dataset_dir}
              onChange={(e) => set("remote_dataset_dir", e.target.value)}
              placeholder="projects/lelab/data   (relative to your home)"
              className="bg-gray-800 border-gray-700 text-white"
            />
          </div>
          <div className="space-y-1.5 sm:col-span-2">
            <Label htmlFor="rs-work" className="text-sm text-gray-300">Work folder on the server</Label>
            <Input
              id="rs-work"
              value={profile.remote_work_dir}
              onChange={(e) => set("remote_work_dir", e.target.value)}
              placeholder="projects/lelab   (where the policy server script goes)"
              className="bg-gray-800 border-gray-700 text-white"
            />
          </div>
        </div>

        {error ? <p className="text-sm text-red-300">{error}</p> : null}
        <div className="flex items-center justify-between gap-2">
          {current ? (
            <Button
              variant="outline"
              size="sm"
              onClick={remove}
              disabled={saving || current.session.in_use}
              className="border-red-500/40 bg-gray-950 text-red-400 hover:bg-red-500/10 hover:text-red-300"
            >
              <Trash2 className="mr-1.5 h-3.5 w-3.5" />
              Remove
            </Button>
          ) : (
            <span />
          )}
          <Button
            onClick={save}
            disabled={saving || !dirty || !name.trim() || !profile.host.trim() || !profile.username.trim()}
            className="bg-blue-600 text-white hover:bg-blue-500"
          >
            {saving ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
            {current ? "Save changes" : "Save server"}
          </Button>
        </div>

        {current && !dirty ? (
          <div className="space-y-3 border-t border-gray-800 pt-4">
            <h3 className="text-sm font-semibold text-gray-200">Connection</h3>
            <ServerConnectForm server={current} allowDisconnect />
            {current.session.connected ? (
              <div className="space-y-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={copyServerScript}
                  disabled={install.busy}
                  className="border-gray-700 bg-gray-950 text-xs text-gray-300 hover:bg-gray-900"
                >
                  {install.busy ? <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> : <Copy className="mr-1.5 h-3.5 w-3.5" />}
                  Copy the policy server script to the work folder
                </Button>
                {install.result ? (
                  <div className="space-y-1">
                    <p className="text-xs text-gray-400">
                      Start it on a GPU node (e.g. in a Slurm job) with the Python you train with:
                    </p>
                    <code className="block break-all rounded bg-black/50 px-2 py-1.5 text-xs text-gray-200">
                      {install.result}
                    </code>
                  </div>
                ) : null}
                {install.error ? <p className="text-xs text-red-300">{install.error}</p> : null}
              </div>
            ) : null}
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
  );
};

export default RemoteServerDialog;
