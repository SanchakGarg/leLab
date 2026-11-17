import { describe, expect, it } from "vitest";
import { defaultVideoLayout, runVideoUrl } from "./runVideo";

describe("defaultVideoLayout", () => {
  it("needs no choice for one camera", () => {
    expect(defaultVideoLayout([])).toEqual({ main: null, corner: null, obvious: true });
    expect(defaultVideoLayout(["front"])).toEqual({ main: "front", corner: null, obvious: true });
  });

  it("puts the wrist camera in the corner when there are two", () => {
    expect(defaultVideoLayout(["wrist", "front"])).toEqual({ main: "front", corner: "wrist", obvious: true });
  });

  it("recognises a wrist camera by the robot camera bound to it", () => {
    const bound: Record<string, string> = { cam_1: "Overhead", cam_2: "Gripper cam" };
    expect(defaultVideoLayout(["cam_1", "cam_2"], (c) => bound[c])).toEqual({
      main: "cam_1",
      corner: "cam_2",
      obvious: true,
    });
  });

  it("asks when there are more than two cameras, or no wrist camera to tell apart", () => {
    expect(defaultVideoLayout(["top", "side", "wrist"])).toEqual({ main: "top", corner: "wrist", obvious: false });
    expect(defaultVideoLayout(["left", "right"])).toEqual({ main: "left", corner: null, obvious: false });
  });
});

it("builds the video URL", () => {
  expect(runVideoUrl("http://localhost:8000", "a b.mp4")).toBe("http://localhost:8000/run-videos/a%20b.mp4");
});
