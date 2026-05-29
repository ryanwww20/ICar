#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Argoverse 2 Sensor Dataset partial downloader
# ============================================================
#
# Usage examples:
#
# 1. List available logs:
#   bash download_av2_logs.sh --list-only
#
# 2. Download first 2 logs, stereo cameras only:
#   bash download_av2_logs.sh --num-logs 2 --mode stereo
#
# 3. Download first 2 full logs:
#   bash download_av2_logs.sh --num-logs 2 --mode full
#
# 4. Save available log IDs to file:
#   bash download_av2_logs.sh --save-log-list av2_val_logs.txt
#
# 5. Download from a prepared log list:
#   bash download_av2_logs.sh --log-list av2_val_logs.txt --num-logs 3 --mode stereo
#
# Requirements:
#   conda install -c conda-forge s5cmd -y
# ============================================================

# -----------------------------
# Default settings
# -----------------------------
SPLIT="val"
OUT_DIR="$HOME/data/av2/sensor"
NUM_LOGS=1
MODE="stereo"      # options: stereo, full
LIST_ONLY=false
SAVE_LOG_LIST=""
LOG_LIST_FILE=""

S3_ROOT="s3://argoverse/datasets/av2/sensor"

# -----------------------------
# Parse arguments
# -----------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --num-logs)
      NUM_LOGS="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --list-only)
      LIST_ONLY=true
      shift 1
      ;;
    --save-log-list)
      SAVE_LOG_LIST="$2"
      shift 2
      ;;
    --log-list)
      LOG_LIST_FILE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage:"
      echo "  bash download_av2_logs.sh [--split val|train|test] [--out-dir PATH] [--num-logs N] [--mode stereo|full]"
      echo ""
      echo "Options:"
      echo "  --list-only              Only list available LOG_IDs"
      echo "  --save-log-list FILE     Save available LOG_IDs to FILE"
      echo "  --log-list FILE          Read LOG_IDs from FILE instead of querying S3"
      echo "  --mode stereo|full       stereo = only stereo cameras + calibration; full = entire log"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# -----------------------------
# Checks
# -----------------------------
if ! command -v s5cmd >/dev/null 2>&1; then
  echo "Error: s5cmd not found."
  echo "Install it with:"
  echo "  conda install -c conda-forge s5cmd -y"
  exit 1
fi

if [[ "$MODE" != "stereo" && "$MODE" != "full" ]]; then
  echo "Error: --mode must be either 'stereo' or 'full'"
  exit 1
fi

if ! [[ "$NUM_LOGS" =~ ^[0-9]+$ ]]; then
  echo "Error: --num-logs must be a positive integer"
  exit 1
fi

# -----------------------------
# Function: get LOG_ID list
# -----------------------------
get_log_ids_from_s3() {
  local s3_path="$S3_ROOT/$SPLIT/"

  echo "Querying available logs from:" >&2
  echo "  $s3_path" >&2
  echo "" >&2

  s5cmd --no-sign-request ls "$s3_path" \
    | awk '$1 == "DIR" {print $2}' \
    | sed 's:/$::'
}

get_log_ids_from_file() {
  local file="$1"

  if [[ ! -f "$file" ]]; then
    echo "Error: log list file not found: $file"
    exit 1
  fi

  # Remove empty lines, comments, and trailing slash; keep UUID-shaped log IDs only.
  grep -v '^\s*$' "$file" \
    | grep -v '^\s*#' \
    | grep -E '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$' \
    | sed 's:/$::'
}

# -----------------------------
# Load LOG_IDs
# -----------------------------
if [[ -n "$LOG_LIST_FILE" ]]; then
  mapfile -t LOG_IDS < <(get_log_ids_from_file "$LOG_LIST_FILE")
else
  mapfile -t LOG_IDS < <(get_log_ids_from_s3)
fi

if [[ "${#LOG_IDS[@]}" -eq 0 ]]; then
  echo "Error: no LOG_ID found."
  exit 1
fi

# -----------------------------
# Save log list if requested
# -----------------------------
if [[ -n "$SAVE_LOG_LIST" ]]; then
  printf "%s\n" "${LOG_IDS[@]}" > "$SAVE_LOG_LIST"
  echo "Saved ${#LOG_IDS[@]} LOG_IDs to:"
  echo "  $SAVE_LOG_LIST"
fi

# -----------------------------
# List only
# -----------------------------
if [[ "$LIST_ONLY" == true ]]; then
  echo "Available LOG_IDs under $S3_ROOT/$SPLIT/:"
  printf "%s\n" "${LOG_IDS[@]}"
  echo ""
  echo "Total logs: ${#LOG_IDS[@]}"
  exit 0
fi

# -----------------------------
# Clip NUM_LOGS
# -----------------------------
if (( NUM_LOGS > ${#LOG_IDS[@]} )); then
  echo "Warning: --num-logs=$NUM_LOGS is larger than available logs=${#LOG_IDS[@]}"
  echo "Will download all ${#LOG_IDS[@]} logs."
  NUM_LOGS="${#LOG_IDS[@]}"
fi

mkdir -p "$OUT_DIR/$SPLIT"

echo "============================================================"
echo "Argoverse 2 Sensor Dataset Downloader"
echo "SPLIT        = $SPLIT"
echo "OUT_DIR      = $OUT_DIR"
echo "NUM_LOGS     = $NUM_LOGS"
echo "MODE         = $MODE"
echo "S3_ROOT      = $S3_ROOT"
echo "TOTAL LOGS   = ${#LOG_IDS[@]}"
echo "============================================================"

# -----------------------------
# Download functions
# -----------------------------
download_full_log() {
  local log_id="$1"
  local src="$S3_ROOT/$SPLIT/$log_id/*"
  local dst="$OUT_DIR/$SPLIT/$log_id/"

  echo ""
  echo "[FULL] Downloading log: $log_id"
  echo "SRC: $src"
  echo "DST: $dst"

  mkdir -p "$dst"

  s5cmd --no-sign-request cp "$src" "$dst"
}

download_stereo_log() {
  local log_id="$1"
  local base_src="$S3_ROOT/$SPLIT/$log_id"
  local base_dst="$OUT_DIR/$SPLIT/$log_id"

  echo ""
  echo "[STEREO] Downloading log: $log_id"
  echo "SRC base: $base_src"
  echo "DST base: $base_dst"

  mkdir -p "$base_dst"

  # -----------------------------
  # Stereo camera images
  # -----------------------------
  mkdir -p "$base_dst/sensors/cameras/stereo_front_left"
  mkdir -p "$base_dst/sensors/cameras/stereo_front_right"

  echo "Downloading stereo_front_left..."
  s5cmd --no-sign-request cp \
    "$base_src/sensors/cameras/stereo_front_left/*" \
    "$base_dst/sensors/cameras/stereo_front_left/"

  echo "Downloading stereo_front_right..."
  s5cmd --no-sign-request cp \
    "$base_src/sensors/cameras/stereo_front_right/*" \
    "$base_dst/sensors/cameras/stereo_front_right/"

  # -----------------------------
  # Calibration files
  # -----------------------------
  echo "Downloading calibration..."
  mkdir -p "$base_dst/calibration"

  s5cmd --no-sign-request cp \
    "$base_src/calibration/*" \
    "$base_dst/calibration/"

  # -----------------------------
  # Ego vehicle pose
  # -----------------------------
  echo "Downloading city_SE3_egovehicle.feather..."
  s5cmd --no-sign-request cp \
    "$base_src/city_SE3_egovehicle.feather" \
    "$base_dst/"
}

# -----------------------------
# Main loop
# -----------------------------
for (( i=0; i<NUM_LOGS; i++ )); do
  LOG_ID="${LOG_IDS[$i]}"

  echo ""
  echo "------------------------------------------------------------"
  echo "[$((i+1))/$NUM_LOGS] LOG_ID = $LOG_ID"
  echo "------------------------------------------------------------"

  if [[ "$MODE" == "full" ]]; then
    download_full_log "$LOG_ID"
  else
    download_stereo_log "$LOG_ID"
  fi
done

echo ""
echo "============================================================"
echo "Done."
echo "Downloaded logs are under:"
echo "  $OUT_DIR/$SPLIT"
echo "============================================================"