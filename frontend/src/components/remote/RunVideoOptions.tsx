import React from "react";
import { Video } from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { RunVideo } from "@/hooks/useRunVideo";

const NO_CORNER = "__none__";

interface Props {
  id: string;
  video: RunVideo;
  /** "run" or "session": when the video can be watched. */
  what?: string;
}

/** "Record a video" with the camera picker, when the layout isn't obvious. */
const RunVideoOptions: React.FC<Props> = ({ id, video, what = "run" }) => {
  const { cameras, recordVideo, defaults, main, corner, cameraLabel } = video;
  return (
    <div className="space-y-3 rounded-md border border-gray-800 p-3">
      <div className="flex items-start gap-3">
        <Checkbox
          id={id}
          checked={recordVideo && cameras.length > 0}
          disabled={cameras.length === 0}
          onCheckedChange={(value) => video.setRecordVideo(value === true)}
          className="mt-0.5 border-gray-500 data-[state=checked]:bg-emerald-500 data-[state=checked]:border-emerald-500"
        />
        <div className="space-y-1">
          <Label htmlFor={id} className="flex items-center gap-1.5 text-sm font-medium text-gray-200 cursor-pointer">
            <Video className="h-4 w-4" />
            Record a video of the {what}
          </Label>
          <p className="text-xs text-gray-500">
            {cameras.length === 0
              ? "This model uses no camera, so there is nothing to record."
              : recordVideo && defaults.obvious && main
                ? `${cameraLabel(main)} full frame${corner ? `, ${cameraLabel(corner)} in the corner` : ""}. Saved on this computer; you can watch it when the ${what} ends.`
                : `Saved on this computer as an mp4; you can watch it when the ${what} ends.`}
          </p>
        </div>
      </div>
      {recordVideo && cameras.length > 0 && !defaults.obvious ? (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label className="text-xs text-gray-400">Full view</Label>
            <Select
              value={main ?? ""}
              onValueChange={(next) => video.choose({ main: next, corner: corner === next ? null : corner })}
            >
              <SelectTrigger aria-label="Full view camera" className="bg-gray-800 border-gray-700 text-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-gray-900 border-gray-700 text-white">
                {cameras.map((camera) => (
                  <SelectItem key={camera} value={camera}>
                    {cameraLabel(camera)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs text-gray-400">In the corner</Label>
            <Select
              value={corner ?? NO_CORNER}
              onValueChange={(next) =>
                video.choose({ main: main ?? cameras[0], corner: next === NO_CORNER ? null : next })
              }
            >
              <SelectTrigger aria-label="Corner camera" className="bg-gray-800 border-gray-700 text-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-gray-900 border-gray-700 text-white">
                <SelectItem value={NO_CORNER}>None</SelectItem>
                {cameras
                  .filter((camera) => camera !== main)
                  .map((camera) => (
                    <SelectItem key={camera} value={camera}>
                      {cameraLabel(camera)}
                    </SelectItem>
                  ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      ) : null}
    </div>
  );
};

export default RunVideoOptions;
