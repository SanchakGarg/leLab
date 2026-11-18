<h1 align="center">🦾 LeLab</h1>

<p align="center">
  <b>The official graphical interface for <a href="https://github.com/huggingface/lerobot">LeRobot</a>.</b>
</p>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/leLab/blob/main/LICENSE)
[![HF Space](https://img.shields.io/badge/🤗-Open%20in%20Spaces-yellow)](https://huggingface.co/spaces/lerobot/LeLab)
[![Discord](https://img.shields.io/badge/Discord-Join_Us-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/q8Dzzpym3f)

</div>

**LeLab** is a web app that puts the full LeRobot workflow — calibrate, teleoperate, record, train, replay — into a single browser UI. Plug in your arm, open the app, and go. No CLI gymnastics, no keyboard prompts.

🤗 A web-native front door to LeRobot, designed so newcomers can get from "unboxing" to "training their first policy" in minutes.

🤗 Install and run everything with a single command.

## Quick Start

Grab the one-liner from the [LeLab Space](https://huggingface.co/spaces/lerobot/LeLab) — it installs and runs LeLab + LeRobot in a single command.

A page will automatically open in your browser and you are ready to go.

## What you can do

<div align="center">
  <table>
    <tr>
      <td>🎯 <b>Calibrate</b></td>
      <td>Guided web flow for both arms — no keyboard prompts.</td>
    </tr>
    <tr>
      <td>🕹️ <b>Teleoperate</b></td>
      <td>Move the leader, the follower mirrors it. Live joint streaming.</td>
    </tr>
    <tr>
      <td>📹 <b>Record</b></td>
      <td>Capture episodes into a LeRobotDataset, with cameras.</td>
    </tr>
    <tr>
      <td>🔎 <b>Browse</b></td>
      <td>Watch the episodes of a dataset on disk — every camera, frame by frame, with a joint-motion trace. No upload required.</td>
    </tr>
    <tr>
      <td>🧠 <b>Train</b></td>
      <td>Kick off a LeRobot training job, watch logs live.</td>
    </tr>
    <tr>
      <td>🤖 <b>Run inference</b></td>
      <td>Execute a trained policy on the follower.</td>
    </tr>
    <tr>
      <td>⏪ <b>Replay</b></td>
      <td>Re-run any recorded episode.</td>
    </tr>
    <tr>
      <td>☁️ <b>Upload</b></td>
      <td>Push your dataset to the <a href="https://huggingface.co/">Hugging Face Hub</a> in one click.</td>
    </tr>
    <tr>
      <td>🖥️ <b>Your GPU server</b></td>
      <td>Send datasets to your own server over SFTP, train there, and run the served model on this arm. See <a href="REMOTE_SERVER.md">REMOTE_SERVER.md</a>.</td>
    </tr>
  </table>
</div>

## Your own GPU server

Train on a GPU machine you reach over SSH, such as a workstation, a cloud VM or a Slurm cluster:

1. Click the server chip in the top bar and enter your user name and the server's IP.
2. Tick **Send to server when done** when you record. The dataset goes to the server over SFTP.
3. Train there with `lerobot-train`.
4. Start `lelab_policy_server.py` with the trained checkpoint. LeLab copies the script for you.
5. In **Jobs → Run model on server**, enter the IP and port its log prints. LeLab tunnels through your SSH login, and the model drives the real follower from the server. Tick **Record a video of the run** to save an mp4 with the wrist camera in a corner.
6. Or tick **Interactive session** to type commands one at a time. Each one runs for up to 60 s, and the page and the video show whether it was done. This also works with local checkpoints.
7. Tick **Blend chunks** to smooth the switch from one predicted action chunk to the next. Both start dialogs remember your settings for the next run.

Your password is never saved. The step-by-step guide, a Slurm example and the safety limits are in [REMOTE_SERVER.md](REMOTE_SERVER.md).

## Noisy USB links

A read or write of an arm that fails now and then (interference, a long cable, a busy hub) no longer ends teleoperation, recording, replay or a model run. LeLab reuses the arm's last positions for that step and carries on, and shows a warning while an arm isn't answering. It stops only after an arm has been silent for 10 s. While recording, a warning also tells you when a gap hit the current episode, so you can re-record it.

## Camera rotation

If a camera is mounted upside down or sideways, select its **Rotation** in the camera setup on the Calibration page.
Choose 0°, 90°, 180°, or 270° clockwise. For an upside-down wrist camera, choose 180°.

LeLab saves the rotation for each camera. Setup and teleoperation previews, recorded frames, and inference use the selected rotation.
Profiles without a rotation setting use 0°.

The resolution setting describes the camera's capture resolution. A 90° or 270° rotation swaps the output dimensions.
For example, a 640 × 480 capture becomes a 480 × 640 observation. The preview shows the complete rotated frame.

Before inference, select the same rotation used for the training dataset. Check it again if you change the camera binding.
Checkpoint image dimensions do not establish image orientation. For example, 0° and 180° have the same dimensions.
The setting applies to new observations; it does not change existing recordings or checkpoints.

## Resources

- **[LeRobot](https://github.com/huggingface/lerobot):** the underlying library — go here for everything beyond the UI.
- **[LeLab Space](https://huggingface.co/spaces/lerobot/LeLab):** try the UI in your browser.
- **[Discord](https://discord.gg/q8Dzzpym3f):** chat with the LeRobot community.
- **[CLAUDE.md](CLAUDE.md):** architecture rundown for contributors.

## Contribute

PRs welcome. Hot-reload mode for working on the code:

```bash
lelab --dev
```

Vite on `:8080`, uvicorn `--reload` on `:8000`. Installing LeLab builds the frontend bundle, so PRs never include `frontend/dist`.

<div align="center">
<sub>Originally hacked together by <a href="https://www.linkedin.com/posts/nicolas-rabault-_lerobot-hackathon-lerobot-ugcPost-7341065019368828930-jTnl/">Team LeLab at the 2025 LeRobot Worldwide Hackathon 🏆</a>, now maintained by the <a href="https://huggingface.co/lerobot">LeRobot</a> team at <a href="https://huggingface.co">Hugging Face</a> with ❤️</sub>
</div>
