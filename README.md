# Monopoly Twin – Physical-to-Digital Board Game System

A real-time hybrid system that bridges a **physical Monopoly board** with a **digital Unreal Engine simulation**, using computer vision, state synchronization, and low-latency communication.

## Demo
Video demo here: https://youtu.be/2GvdDb1VBgk

## Overview

This project explores how physical interactions can be captured, interpreted, and reflected inside a real-time 3D environment.

Current capabilities:

* Pawn detection via OpenCV
* Board position mapping
* WebSocket communication pipeline
* Real-time sync with Unreal Engine
* Virtual dice system in Unreal
* Camera-based pawn tracking

## Why This Project Exists

Most board game digitization stops at UI replication.

This project focuses on:

* **Physical-to-digital bridging**
* **Real-time synchronization challenges**
* **System modularity for extensibility**
* **Low-latency event pipelines**

It is designed as a **foundation for mixed-reality gameplay systems**.

---

## System Architecture

```
[Camera Input]
      ↓
[Vision Pipeline (OpenCV)]
      ↓
[Game State Engine]
      ↓
[WebSocket Server]
      ↓
[Unreal Engine Client]
      ↓
[3D Visualization + Logic]
```

Key design principles:

* Decoupled modules (Vision / State / Networking)
* Stateless message protocol
* Real-time event-driven updates
* Engine-agnostic backend

---

## Tech Stack

* **Python** – Vision + server
* **OpenCV** – Pawn detection & tracking
* **WebSockets** – Real-time communication
* **Unreal Engine 5** – Visualization & interaction layer
* **HTML/JS** – Debug & control UI

---

## Communication Model

All interactions are driven via a structured message protocol:

```json
{
  "type": "PAWN_UPDATE",
  "player_id": "P1",
  "board_index": 12,
  "timestamp": 1710000000
}
```

Design goals:

* Deterministic state sync
* Minimal payload size
* Engine-independent messaging

---

## Demo

See `assets/demo.mp4` for current progress:

* Pawn tracking
* Board mapping
* Unreal sync
* Dice interaction

---

## Project Structure

* `backend/` → Vision + server logic
* `unreal/` → Engine integration
* `client/` → Debug UI
* `docs/` → System design & decisions

---

## Current Challenges

* Robust detection under lighting variation
* Multi-pawn tracking consistency
* Latency smoothing for real-time sync
* Error handling in physical interaction

---

## Roadmap

* Multi-player support
* Banker control system
* Game rule engine
* Mobile companion app
* AR overlay experimentation

---

## About Me

Unreal Engine developer with 5+ years in virtual production and real-time systems.

This project demonstrates:

* System design thinking
* Cross-domain integration (CV + networking + UE)
* Real-time pipeline architecture
* Ownership of end-to-end implementation

---

## Feedback / Collaboration

Open to discussions on:

* Mixed reality systems
* Unreal integrations
* Real-time architecture design
