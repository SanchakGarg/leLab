// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import RemoteServerDialog from "./RemoteServerDialog";
import { RemoteServerProvider } from "@/contexts/RemoteServerContext";
import { button, click, json, type } from "./testUtils";

const mocks = vi.hoisted(() => ({ fetcher: vi.fn() }));
vi.mock("@/contexts/ApiContext", () => ({ useApi: () => ({ baseUrl: "http://test", fetchWithHeaders: mocks.fetcher }) }));
vi.mock("@/components/ui/select", async () => (await import("./testUtils")).uiMocks.select());
vi.mock("@/components/ui/dialog", async () => (await import("./testUtils")).uiMocks.dialog());

type Server = Record<string, unknown> & { name: string; session: { connected: boolean } };
let servers: Server[];
let calls: { method: string; path: string; body: Record<string, unknown> | null }[];
let root: Root;
let container: HTMLDivElement;

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  container = document.createElement("div");
  root = createRoot(container);
  servers = [];
  calls = [];
  mocks.fetcher.mockReset();
  mocks.fetcher.mockImplementation(async (url: string, init: RequestInit = {}) => {
    const method = init.method ?? "GET";
    const path = url.replace("http://test", "");
    const body = init.body ? JSON.parse(String(init.body)) : null;
    calls.push({ method, path, body });
    if (path === "/remote-servers") return json({ servers });
    if (method === "PUT" && path === "/remote-servers/cluster") {
      servers = [{ ...body, name: "cluster", session: { connected: false } }];
      return json(body);
    }
    if (path === "/remote-servers/cluster/connect") {
      if (!body.trust_fingerprint) {
        return json(
          { detail: { message: "10.2.100.14 is not a known host yet.", code: "unknown_host_key", fingerprint: "SHA256:abc", key_type: "ssh-ed25519" } },
          409,
        );
      }
      servers = [{ ...servers[0], session: { connected: true } }];
      return json({ connected: true });
    }
    return json({}, 404);
  });
});
afterEach(() => act(() => root.unmount()));

async function render() {
  await act(async () =>
    root.render(
      <RemoteServerProvider>
        <RemoteServerDialog open onOpenChange={vi.fn()} />
      </RemoteServerProvider>,
    ),
  );
}

const input = (id: string) => container.querySelector<HTMLInputElement>(`#${id}`)!;

test("a saved profile never carries the password; connecting sends it and confirms the host key", async () => {
  await render();
  type(input("rs-host"), "10.2.100.14");
  type(input("rs-user"), "student");
  type(input("rs-data"), "projects/lelab/data");
  await click(button(container, "Save server"));

  const put = calls.find((c) => c.method === "PUT")!;
  expect(put.path).toBe("/remote-servers/cluster");
  expect(put.body).toMatchObject({ host: "10.2.100.14", username: "student", auth: "password", port: 22 });
  expect(Object.keys(put.body!)).not.toContain("password");

  type(input("pw-cluster"), "hunter2");
  await click(button(container, "Connect"));
  const first = calls.filter((c) => c.path.endsWith("/connect"));
  expect(first).toHaveLength(1);
  expect(first[0].body).toEqual({ password: "hunter2" });
  expect(container.textContent).toContain("SHA256:abc");

  await click(button(container, "Trust and connect"));
  const second = calls.filter((c) => c.path.endsWith("/connect"))[1];
  expect(second.body).toEqual({ password: "hunter2", trust_fingerprint: "SHA256:abc" });
  expect(container.textContent).toContain("Connected to student@10.2.100.14");
  // The password went only to /connect.
  expect(calls.filter((c) => JSON.stringify(c.body ?? {}).includes("hunter2")).every((c) => c.path.endsWith("/connect"))).toBe(true);
});
