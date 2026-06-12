"""Automotive-style timing breakdown for VGGT perception pipelines."""

from __future__ import annotations

# One-time startup (cold start at ignition / process launch).
TIMING_MODEL_LOAD = "model_load"

# Per-frame steady-state pipeline.
TIMING_PREPROCESS = "preprocess"
TIMING_INFERENCE = "inference"
TIMING_DECODE = "decode"
TIMING_POINTCLOUD_EXPORT = "pointcloud_export"
TIMING_POST_PROCESS = "post_process"

PER_FRAME_KEYS = (
    TIMING_PREPROCESS,
    TIMING_INFERENCE,
    TIMING_DECODE,
    TIMING_POINTCLOUD_EXPORT,
    TIMING_POST_PROCESS,
)


def empty_startup_timings() -> dict[str, float]:
    return {TIMING_MODEL_LOAD: 0.0}


def empty_frame_timings() -> dict[str, float]:
    return {key: 0.0 for key in PER_FRAME_KEYS}


def online_perception_total(frame: dict[str, float]) -> float:
    """Preprocess + inference + decode (typical on-vehicle online loop)."""
    return (
        frame.get(TIMING_PREPROCESS, 0.0)
        + frame.get(TIMING_INFERENCE, 0.0)
        + frame.get(TIMING_DECODE, 0.0)
    )


def perception_total(frame: dict[str, float]) -> float:
    return (
        frame.get(TIMING_PREPROCESS, 0.0)
        + frame.get(TIMING_INFERENCE, 0.0)
        + frame.get(TIMING_DECODE, 0.0)
        + frame.get(TIMING_POINTCLOUD_EXPORT, 0.0)
    )


def end_to_end_total(frame: dict[str, float]) -> float:
    return perception_total(frame) + frame.get(TIMING_POST_PROCESS, 0.0)


def average_frames(frames: list[dict[str, float]]) -> dict[str, float]:
    if not frames:
        return empty_frame_timings()
    n = len(frames)
    avg = empty_frame_timings()
    for key in PER_FRAME_KEYS:
        avg[key] = sum(f.get(key, 0.0) for f in frames) / n
    return avg


def _ms(seconds: float) -> float:
    return seconds * 1000.0


def _hz(latency_s: float) -> float | None:
    if latency_s <= 0:
        return None
    return 1.0 / latency_s


def _fmt_ms(seconds: float, width: int = 8) -> str:
    return f"{_ms(seconds):{width}.1f} ms"


def _fmt_hz(seconds: float) -> str:
    hz = _hz(seconds)
    if hz is None:
        return "  n/a"
    return f"{hz:5.2f} Hz"


def _budget_line(label: str, budget_ms: float, latency_s: float) -> str:
    latency_ms = _ms(latency_s)
    status = "MET" if latency_ms <= budget_ms else "NOT MET"
    return f"  {label} ({budget_ms:.0f} ms budget): {status} ({latency_ms:.1f} ms)"


def print_automotive_timing_analysis(
    startup: dict[str, float],
    frames: list[dict[str, float]],
    *,
    label: str = "",
) -> None:
    """Print timing suited for on-vehicle deployment reports.

    Model load is reported once at startup. Per-frame numbers exclude startup and
    reflect the steady-state perception loop (preprocess -> inference -> export).
    """
    header = "=== Timing analysis (on-vehicle deployment model)"
    if label:
        header += f" — {label}"
    header += " ==="
    print(f"\n{header}")
    print(
        "  Assumption: model loaded once at startup; per-frame timings are "
        "steady-state (exclude one-time model load)."
    )

    load_s = startup.get(TIMING_MODEL_LOAD, 0.0)
    if load_s > 0:
        print("\n[A] One-time startup (cold start)")
        print(f"  Model load: {_fmt_ms(load_s, width=10)}")

    if not frames:
        print("\n  (no per-frame timings)")
        return

    if len(frames) == 1:
        summary = frames[0]
        section = "[B] Per-frame pipeline"
    else:
        summary = average_frames(frames)
        section = (
            f"[B] Per-frame pipeline (average over {len(frames)} frames)"
        )
        if len(frames) >= 2:
            steady = average_frames(frames[1:])
            steady_e2e = end_to_end_total(steady)
            print(
                f"\n  Note: steady-state avg (frames 1..{len(frames) - 1}, "
                f"excl. warmup frame 0): "
                f"end-to-end {_fmt_ms(steady_e2e)}, {_fmt_hz(steady_e2e)}"
            )

    online_s = online_perception_total(summary)
    perc_s = perception_total(summary)
    e2e_s = end_to_end_total(summary)

    print(f"\n{section}")
    print(f"  1. Camera preprocess:  {_fmt_ms(summary.get(TIMING_PREPROCESS, 0.0))}")
    print(f"  2. VGGT inference:     {_fmt_ms(summary.get(TIMING_INFERENCE, 0.0))}")
    print(f"  3. Output decode:      {_fmt_ms(summary.get(TIMING_DECODE, 0.0))}")
    print(
        f"     Online perception: {_fmt_ms(online_s)}  "
        f"({_fmt_hz(online_s).strip()})"
    )
    print(
        f"  4. Point cloud export: {_fmt_ms(summary.get(TIMING_POINTCLOUD_EXPORT, 0.0))}"
    )
    print(
        f"     Perception subtotal: {_fmt_ms(perc_s)}  "
        f"({_fmt_hz(perc_s).strip()})"
    )
    print(f"  5. Post process:       {_fmt_ms(summary.get(TIMING_POST_PROCESS, 0.0))}")
    print(
        f"     End-to-end:          {_fmt_ms(e2e_s)}  "
        f"({_fmt_hz(e2e_s).strip()})"
    )

    print("\n[C] Real-time feasibility (typical automotive budgets)")
    print(_budget_line("Online perception @ 10 Hz", 100.0, online_s))
    print(_budget_line("Online perception @ 30 Hz", 33.0, online_s))
    print(_budget_line("Perception @ 10 Hz", 100.0, perc_s))
    print(_budget_line("End-to-end @ 10 Hz", 100.0, e2e_s))


def timing_to_metadata(
    startup: dict[str, float],
    frames: list[dict[str, float]],
) -> dict:
    """Structured timing payload for metadata.json / run_summary.json."""
    frame_payloads = []
    for i, frame in enumerate(frames):
        perc = perception_total(frame)
        online = online_perception_total(frame)
        e2e = end_to_end_total(frame)
        frame_payloads.append(
            {
                "frame_index": i,
                **{k: frame.get(k, 0.0) for k in PER_FRAME_KEYS},
                "online_perception_s": online,
                "perception_total_s": perc,
                "end_to_end_s": e2e,
                "online_perception_hz": _hz(online),
                "perception_hz": _hz(perc),
                "end_to_end_hz": _hz(e2e),
            }
        )

    payload: dict = {
        "model": "on_vehicle_deployment",
        "startup_s": dict(startup),
        "frames": frame_payloads,
    }
    if frames:
        avg = average_frames(frames)
        payload["per_frame_avg_s"] = {
            **{k: avg.get(k, 0.0) for k in PER_FRAME_KEYS},
            "online_perception_s": online_perception_total(avg),
            "perception_total_s": perception_total(avg),
            "end_to_end_s": end_to_end_total(avg),
        }
        if len(frames) >= 2:
            steady = average_frames(frames[1:])
            payload["steady_state_avg_s"] = {
                **{k: steady.get(k, 0.0) for k in PER_FRAME_KEYS},
                "online_perception_s": online_perception_total(steady),
                "perception_total_s": perception_total(steady),
                "end_to_end_s": end_to_end_total(steady),
            }
    return payload
