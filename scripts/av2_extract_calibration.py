#!/usr/bin/env python3
"""Extract camera calibration from Argoverse 2 Sensor Dataset logs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from av2_utils import find_log_dirs, resolve_split_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract AV2 camera calibration for pose evaluation."
    )
    parser.add_argument("--av2-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--log-id", type=str, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/av2_calibration_summary.json"),
    )
    return parser.parse_args()


def _matrix_to_list(value) -> list | None:
    if value is None:
        return None
    try:
        import numpy as np

        arr = np.asarray(value)
        return arr.tolist()
    except Exception:
        return None


def extract_with_av2_api(log_dirs: list[Path]) -> dict | None:
    try:
        from av2.utils.io import read_feather
    except ImportError:
        return None

    summary: dict = {"source": "av2-api", "logs": {}}

    for log_dir in log_dirs:
        log_id = log_dir.name
        log_entry = {"cameras": {}, "notes": []}

        calib_dir = log_dir / "calibration"
        intrinsics_path = calib_dir / "intrinsics.feather"
        extrinsics_path = calib_dir / "extrinsics.feather"

        if intrinsics_path.is_file():
            intrinsics = read_feather(intrinsics_path)
            for row in intrinsics.itertuples(index=False):
                camera_name = getattr(row, "sensor_name", None) or getattr(row, "camera_name", None)
                if camera_name is None:
                    continue
                log_entry["cameras"][str(camera_name)] = {
                    "camera_name": str(camera_name),
                    "intrinsics": _matrix_to_list(getattr(row, "K", None)),
                    "extrinsics": None,
                    "image_size": [
                        int(getattr(row, "width_px", 0) or 0),
                        int(getattr(row, "height_px", 0) or 0),
                    ],
                }

        if extrinsics_path.is_file():
            extrinsics = read_feather(extrinsics_path)
            for row in extrinsics.itertuples(index=False):
                camera_name = getattr(row, "sensor_name", None) or getattr(row, "camera_name", None)
                if camera_name is None:
                    continue
                camera_name = str(camera_name)
                if camera_name not in log_entry["cameras"]:
                    log_entry["cameras"][camera_name] = {"camera_name": camera_name}
                log_entry["cameras"][camera_name]["extrinsics"] = _matrix_to_list(
                    getattr(row, "ego_SE3_sensor", None) or getattr(row, "extrinsic", None)
                )

        if not log_entry["cameras"]:
            log_entry["notes"].append("av2-api available but no calibration rows found in feather files")
        summary["logs"][log_id] = log_entry

    return summary if summary["logs"] else None


def _load_json_file(path: Path) -> dict | list | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_with_file_search(log_dirs: list[Path]) -> dict:
    summary: dict = {"source": "file-search-fallback", "logs": {}}
    patterns = (
        "**/calibration/**/*.json",
        "**/*intrinsic*.json",
        "**/*extrinsic*.json",
        "**/*calibration*.json",
        "**/intrinsics.feather",
        "**/extrinsics.feather",
    )

    for log_dir in log_dirs:
        log_id = log_dir.name
        log_entry = {"cameras": {}, "files_found": [], "notes": []}

        for pattern in patterns:
            for path in log_dir.glob(pattern):
                log_entry["files_found"].append(str(path.resolve()))
                if path.suffix.lower() == ".json":
                    payload = _load_json_file(path)
                    if isinstance(payload, dict):
                        for key, value in payload.items():
                            if not isinstance(value, dict):
                                continue
                            log_entry["cameras"].setdefault(
                                str(key),
                                {
                                    "camera_name": str(key),
                                    "intrinsics": value.get("intrinsics") or value.get("K"),
                                    "extrinsics": value.get("extrinsics") or value.get("extrinsic"),
                                    "image_size": value.get("image_size") or value.get("resolution"),
                                },
                            )

        if not log_entry["cameras"] and not log_entry["files_found"]:
            log_entry["notes"].append(
                "No calibration json/feather files found. Install av2-api or verify log download completeness."
            )
        summary["logs"][log_id] = log_entry

    return summary


def main() -> int:
    args = parse_args()
    av2_root = args.av2_root.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    if not av2_root.exists():
        print(f"ERROR: --av2-root does not exist: {av2_root}", file=sys.stderr)
        return 1

    split_root = resolve_split_root(av2_root, args.split)
    if args.log_id:
        log_dir = split_root / args.log_id
        if not log_dir.is_dir():
            print(f"ERROR: log-id not found: {log_dir}", file=sys.stderr)
            return 1
        log_dirs = [log_dir.resolve()]
    else:
        log_dirs = find_log_dirs(split_root, max_logs=5)

    if not log_dirs:
        print(f"ERROR: No logs found under {split_root}", file=sys.stderr)
        return 1

    summary = extract_with_av2_api(log_dirs)
    if summary is None:
        print("av2-api not available; using file-search fallback.", file=sys.stderr)
        summary = extract_with_file_search(log_dirs)
    elif any(not entry.get("cameras") for entry in summary.get("logs", {}).values()):
        fallback = extract_with_file_search(log_dirs)
        for log_id, entry in fallback["logs"].items():
            if log_id not in summary["logs"]:
                summary["logs"][log_id] = entry
            elif not summary["logs"][log_id].get("cameras") and entry.get("cameras"):
                summary["logs"][log_id] = entry

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "av2_root": str(av2_root),
        "split": args.split,
        **summary,
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote calibration summary: {output_json.resolve()}")
    for log_id, entry in payload.get("logs", {}).items():
        num_cameras = len(entry.get("cameras", {}))
        print(f"  {log_id}: {num_cameras} camera entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
