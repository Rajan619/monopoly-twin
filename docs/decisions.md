# Key Technical Decisions

## Why WebSockets over REST?

Real-time bidirectional communication is required for gameplay synchronization.

## Why Python for Backend?

Fast iteration for computer vision and prototyping.

## Why Unreal for Visualization?

High-fidelity real-time rendering and interaction system.

## Why Modular Architecture?

Future extensibility:

* Replace OpenCV with ML model
* Replace Unreal with Unity/Web
* Add mobile clients

## Why Not Monolithic Design?

Avoid tight coupling between:

* Vision
* Game logic
* Rendering

---

## Trade-offs

| Decision              | Benefit          | Trade-off                      |
| --------------------- | ---------------- | ------------------------------ |
| WebSockets            | Real-time sync   | Requires connection management |
| OpenCV                | Fast prototyping | Sensitive to lighting          |
| External state engine | Scalable         | More complexity                |
