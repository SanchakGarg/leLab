import { expect, test } from "vitest";
import { imageFeaturesFromHello, PolicyHello } from "./remoteApi";

test("cameras capture at their own resolution, not at the model's input size", () => {
  // smolvla: an older policy server still reports the base model's 256×256 for each camera.
  const hello = {
    type: "hello",
    cameras: ["birdseye", "wrist", "top"],
    image_shapes: { birdseye: [256, 256, 3], wrist: [256, 256, 3], top: [480, 640, 3] },
  } as unknown as PolicyHello;
  const robot = [
    { name: "Birdseye", width: 1280, height: 720 },
    { name: "wrist", width: 640, height: 480 },
  ];

  expect(imageFeaturesFromHello(hello, robot)).toEqual({
    birdseye: { width: 1280, height: 720 },
    wrist: { width: 640, height: 480 },
    top: { width: 640, height: 480 },
  });
});
