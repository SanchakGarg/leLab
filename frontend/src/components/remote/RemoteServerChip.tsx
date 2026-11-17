import React, { useState } from "react";
import { Server } from "lucide-react";
import { useRemoteServers } from "@/contexts/RemoteServerContext";
import RemoteServerDialog from "./RemoteServerDialog";

/** Top-bar chip: the selected remote server and whether LeLab is connected to it. Opens the server settings. */
const RemoteServerChip: React.FC = () => {
  const { selected, loading } = useRemoteServers();
  const [open, setOpen] = useState(false);
  if (loading) return null;

  const connected = !!selected?.session.connected;
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="inline-flex items-center gap-2 rounded-full border border-gray-800 bg-gray-900/60 px-3 py-1 text-xs text-gray-200 hover:bg-gray-800 transition-colors"
        aria-label={selected ? `Remote server ${selected.name}` : "Add a remote server"}
        title={selected ? `${selected.username}@${selected.host}` : "Send datasets to your server and run models from it"}
      >
        {selected ? (
          <>
            <span
              className={`h-2 w-2 rounded-full ${connected ? "bg-emerald-400" : "bg-gray-500"}`}
              aria-hidden="true"
            />
            <span>{selected.name}</span>
          </>
        ) : (
          <>
            <Server className="h-3 w-3" />
            <span>Add server</span>
          </>
        )}
      </button>
      <RemoteServerDialog open={open} onOpenChange={setOpen} />
    </>
  );
};

export default RemoteServerChip;
