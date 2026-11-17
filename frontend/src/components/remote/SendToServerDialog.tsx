import React, { useEffect, useRef, useState } from "react";
import { CheckCircle, Copy, Loader2, Send, XCircle } from "lucide-react";
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
import { useRemoteServers } from "@/contexts/RemoteServerContext";
import { useRemoteUpload } from "@/hooks/useRemoteUpload";
import { formatBytes } from "@/lib/remoteApi";
import RemoteServerDialog from "./RemoteServerDialog";
import ServerConnectForm from "./ServerConnectForm";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  datasetRepoId: string;
  /** Start as soon as the server is connected (after a recording that asked for it). */
  autoStart?: boolean;
}

/** Copy a local dataset to the remote server over SFTP, with progress, and show how to train on it there. */
const SendToServerDialog: React.FC<Props> = ({ open, onOpenChange, datasetRepoId, autoStart = false }) => {
  const { servers, selected, selectServer } = useRemoteServers();
  const { status, error, starting, start, cancel } = useRemoteUpload(open);
  const [remoteDir, setRemoteDir] = useState("");
  const [remoteName, setRemoteName] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const autoStarted = useRef(false);

  const folderName = datasetRepoId.split("/").pop() ?? datasetRepoId;
  useEffect(() => {
    if (!open) return;
    setRemoteDir(selected?.remote_dataset_dir ?? "");
    setRemoteName(folderName);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, selected?.name, folderName]);

  const mine = status.dataset_repo_id === datasetRepoId;
  const running = status.state === "running";
  const connected = !!selected?.session.connected;

  const send = () => {
    if (!selected) return;
    start({ server: selected.name, dataset_repo_id: datasetRepoId, remote_dir: remoteDir, remote_name: remoteName });
  };

  useEffect(() => {
    if (open && autoStart && connected && !autoStarted.current && !running) {
      autoStarted.current = true;
      send();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, autoStart, connected, running]);

  const pct = status.bytes_total ? Math.min(100, (100 * (status.bytes_done ?? 0)) / status.bytes_total) : 0;

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="bg-gray-900 border-gray-800 text-white sm:max-w-[560px] max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-white">
              <Send className="h-5 w-5" />
              Send to server
            </DialogTitle>
            <DialogDescription className="text-gray-400 break-all">
              Copy <strong className="text-gray-200">{datasetRepoId}</strong> to your server over SFTP, to
              train on it there.
            </DialogDescription>
          </DialogHeader>

          {!selected ? (
            <div className="space-y-3">
              <p className="text-sm text-gray-300">No server saved yet.</p>
              <Button onClick={() => setSettingsOpen(true)} className="bg-blue-600 text-white hover:bg-blue-500">
                Add a server
              </Button>
            </div>
          ) : (
            <div className="space-y-4">
              {servers.length > 1 ? (
                <div className="space-y-1.5">
                  <Label className="text-sm text-gray-300">Server</Label>
                  <Select value={selected.name} onValueChange={selectServer} disabled={running}>
                    <SelectTrigger className="bg-gray-800 border-gray-700 text-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-gray-900 border-gray-700 text-white">
                      {servers.map((s) => (
                        <SelectItem key={s.name} value={s.name}>
                          {s.name} — {s.username}@{s.host}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              ) : null}

              <ServerConnectForm server={selected} />

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div className="space-y-1.5">
                  <Label htmlFor="send-dir" className="text-sm text-gray-300">Folder on the server</Label>
                  <Input
                    id="send-dir"
                    value={remoteDir}
                    disabled={running}
                    onChange={(e) => setRemoteDir(e.target.value)}
                    placeholder="home folder"
                    className="bg-gray-800 border-gray-700 text-white"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="send-name" className="text-sm text-gray-300">Dataset folder name</Label>
                  <Input
                    id="send-name"
                    value={remoteName}
                    disabled={running}
                    onChange={(e) => setRemoteName(e.target.value)}
                    className="bg-gray-800 border-gray-700 text-white"
                  />
                </div>
              </div>

              {mine && status.state !== "idle" ? (
                <div className="space-y-2 rounded-md border border-gray-800 bg-black/30 p-3 text-sm">
                  <div className="flex items-center justify-between text-gray-300">
                    <span>
                      {status.files_done ?? 0}/{status.files_total ?? 0} files ·{" "}
                      {formatBytes(status.bytes_done)} of {formatBytes(status.bytes_total)}
                    </span>
                    <span>{pct.toFixed(0)}%</span>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-gray-800">
                    <div
                      className={`h-1.5 rounded-full transition-all ${status.state === "error" ? "bg-red-500" : "bg-emerald-500"}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  {running && status.current_file ? (
                    <p className="truncate text-xs text-gray-500">{status.current_file}</p>
                  ) : null}
                  {status.state === "done" ? (
                    <div className="space-y-2">
                      <p className="flex items-center gap-2 text-emerald-300">
                        <CheckCircle className="h-4 w-4" />
                        On the server at <code className="break-all text-xs">{status.remote_path}</code>
                        {status.files_skipped ? ` (${status.files_skipped} files were already there)` : ""}
                      </p>
                      {status.train_command ? (
                        <div className="space-y-1">
                          <p className="text-xs text-gray-400">Train on it there, for example:</p>
                          <div className="flex items-start gap-2">
                            <code className="block flex-1 break-all rounded bg-black/50 px-2 py-1.5 text-xs text-gray-200">
                              {status.train_command}
                            </code>
                            <Button
                              size="icon"
                              variant="ghost"
                              aria-label="Copy the training command"
                              onClick={() => {
                                navigator.clipboard?.writeText(status.train_command ?? "");
                                setCopied(true);
                              }}
                              className="h-7 w-7 text-gray-400 hover:text-white"
                            >
                              {copied ? <CheckCircle className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                            </Button>
                          </div>
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                  {status.state === "error" ? (
                    <p className="flex items-center gap-2 text-red-300">
                      <XCircle className="h-4 w-4" />
                      {status.error}
                    </p>
                  ) : null}
                  {status.state === "cancelled" ? (
                    <p className="text-gray-400">Cancelled. Sending again resumes where it stopped.</p>
                  ) : null}
                </div>
              ) : null}
              {running && !mine ? (
                <p className="text-xs text-amber-300">Another upload ({status.dataset_repo_id}) is running.</p>
              ) : null}
              {error ? <p className="text-sm text-red-300">{error}</p> : null}

              <div className="flex justify-end gap-2">
                {running && mine ? (
                  <Button variant="outline" onClick={cancel} className="border-gray-700 bg-gray-950 text-gray-300 hover:bg-gray-900">
                    Cancel upload
                  </Button>
                ) : null}
                <Button
                  onClick={send}
                  disabled={!connected || running || starting || !remoteName.trim()}
                  className="bg-emerald-600 text-white hover:bg-emerald-500"
                >
                  {running || starting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Send className="mr-2 h-4 w-4" />}
                  {mine && status.state === "done" ? "Send again" : "Send"}
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
      <RemoteServerDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
    </>
  );
};

export default SendToServerDialog;
