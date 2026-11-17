// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";
import ArmLinkNotice, { type ArmLinkIssue } from "./ArmLinkNotice";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;
let container: HTMLDivElement;

function render(issues: ArmLinkIssue[] | undefined, recording = false) {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => root!.render(<ArmLinkNotice issues={issues} recording={recording} />));
  return container;
}

afterEach(() => {
  act(() => root?.unmount());
  container?.remove();
  root = null;
});

describe("ArmLinkNotice", () => {
  it("renders nothing while both arms answer", () => {
    expect(render([]).innerHTML).toBe("");
    expect(render(undefined).innerHTML).toBe("");
  });

  it("says which arm is silent now and when LeLab gives up", () => {
    const text = render([{ device: "leader arm", ongoing: true, duration_s: 2.34 }]).textContent;
    expect(text).toContain("The leader arm isn't answering (2.3 s)");
    expect(text).toContain("stops after 10 s");
  });

  it("suggests re-recording after a gap during an episode", () => {
    const text = render([{ device: "follower arm", ongoing: false, duration_s: 1.5, skipped: 12 }], true).textContent;
    expect(text).toContain("stopped answering for 1.5 s");
    expect(text).toContain("Re-record the episode");
  });
});
