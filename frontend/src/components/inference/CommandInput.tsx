import React, { useRef, useState } from "react";
import { Loader2, SendHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface Props {
  /** Why a command can't be sent now (another one runs, the arm is going back, ...); null when it can. */
  blockedReason: string | null;
  /** Sends the command; resolves true when it was accepted (the box is then cleared). */
  onSend: (text: string) => Promise<boolean>;
}

/** The box at the bottom of an interactive session: type a command, Enter sends it. */
const CommandInput: React.FC<Props> = ({ blockedReason, onSend }) => {
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const canSend = !blockedReason && !sending && text.trim().length > 0;

  const send = async () => {
    if (!canSend) return;
    setSending(true);
    try {
      if (await onSend(text.trim())) setText("");
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  };

  return (
    <div className="space-y-1.5">
      <div className="flex gap-2">
        <Input
          ref={inputRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              void send();
            }
          }}
          maxLength={500}
          autoFocus
          aria-label="Command"
          placeholder="Tell the robot what to do, then press Enter"
          className="h-11 bg-gray-800 border-gray-700 text-white placeholder:text-gray-500"
        />
        <Button
          onClick={() => void send()}
          disabled={!canSend}
          aria-label="Send command"
          className="h-11 bg-green-500 px-4 text-white hover:bg-green-600 disabled:opacity-40"
        >
          {sending ? <Loader2 className="h-4 w-4 animate-spin" /> : <SendHorizontal className="h-4 w-4" />}
        </Button>
      </div>
      <p className="min-h-4 text-xs text-gray-500">{blockedReason ?? "Enter sends the command."}</p>
    </div>
  );
};

export default CommandInput;
