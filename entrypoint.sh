#!/usr/bin/env bash
set -euo pipefail

############################################################
# Job ID + project layout
############################################################

# Batch sets BATCH_JOB_ID; locally we default to "local"
JOB_ID="${BATCH_JOB_ID:-local}"

# PROJECT_ROOT is where simulation.jl + scenarios.json live and
# where data/input and data/output are rooted.
export PROJECT_ROOT="${PROJECT_ROOT:-/app}"

LOCAL_INPUT_DIR="${PROJECT_ROOT}/data/input"
LOCAL_OUTPUT_ROOT="${PROJECT_ROOT}/data/output"

# Scenario name (maps to a key in scenarios.json)
SCENARIO="${SCENARIO:-example}"
STOP_DAYS="${STOP_DAYS:-365}"

# Validate STOP_DAYS: must be a positive integer (>= 1)
case "${STOP_DAYS}" in
  ''|*[!0-9]*) echo "❌ STOP_DAYS must be a positive integer, got '${STOP_DAYS}'."; exit 1 ;;
esac
[ "${STOP_DAYS}" -ge 1 ] || { echo "❌ STOP_DAYS must be a positive integer (>= 1), got '${STOP_DAYS}'."; exit 1; }

LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_ROOT}/${SCENARIO}"

mkdir -p "${LOCAL_INPUT_DIR}" "${LOCAL_OUTPUT_DIR}"

############################################################
# Logging: tee everything to a log file + console
############################################################

LOG_FILE="${LOCAL_OUTPUT_DIR}/simulation.log"
echo "📄 Logging to ${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

############################################################
# Basic configuration
############################################################

GCS_INPUT_PREFIX="${GCS_INPUT_PREFIX:-input}"
GCS_OUTPUT_PREFIX="${GCS_OUTPUT_PREFIX:-output}"

die() {
  echo "❌ $*"
  exit 1
}

# Which Julia script to run (can be overridden via env)
SCRIPT_PATH="${SIMULATION_LAUNCHER:-/app/simulation.jl}"

echo "=== FjordSim / Oceananigans entrypoint ==="
echo "  JOB_ID          = ${JOB_ID}"
echo "  SCENARIO        = ${SCENARIO}"
echo "  STOP_DAYS       = ${STOP_DAYS}"
echo "  SCRIPT_PATH     = ${SCRIPT_PATH}"
echo "  PROJECT_ROOT    = ${PROJECT_ROOT}"
echo "  LOCAL_INPUT_DIR = ${LOCAL_INPUT_DIR}"
echo "  LOCAL_OUTPUT_DIR= ${LOCAL_OUTPUT_DIR}"

############################################################
# Mode selection: local vs cloud
############################################################

if [ -z "${GCS_BUCKET:-}" ]; then
  echo "🌍 Local development mode (no GCS_BUCKET set)"
  echo "  Input  → ${LOCAL_INPUT_DIR}"
  echo "  Output → ${LOCAL_OUTPUT_DIR}"
else
  echo "☁️  Cloud mode (GCS_BUCKET=${GCS_BUCKET})"
  echo "  Input prefix:  gs://${GCS_BUCKET}/${GCS_INPUT_PREFIX}/"
  echo "  Output prefix: gs://${GCS_BUCKET}/${GCS_OUTPUT_PREFIX}/${JOB_ID}/"

  echo "=== Downloading input from GCS → ${LOCAL_INPUT_DIR}"
  gsutil -m cp -r "gs://${GCS_BUCKET}/${GCS_INPUT_PREFIX}/*" "${LOCAL_INPUT_DIR}" || \
    echo "⚠️ No input files found at gs://${GCS_BUCKET}/${GCS_INPUT_PREFIX}/ (continuing)."
fi

############################################################
# Verify input contents (if any)
############################################################

echo "=== Verifying input files in ${LOCAL_INPUT_DIR}"
if [ -d "${LOCAL_INPUT_DIR}" ] && [ "$(ls -A "${LOCAL_INPUT_DIR}" 2>/dev/null)" ]; then
  echo "📂 Contents of ${LOCAL_INPUT_DIR}:"
  find "${LOCAL_INPUT_DIR}" -maxdepth 2 -type f -exec ls -lh {} \;
  echo "✅ Input directory is ready."
else
  die "${LOCAL_INPUT_DIR} is empty or missing. Check your local input mount or GCS paths."
fi

############################################################
# Architecture selection and validation
############################################################

ARCH_EXPECTED_RAW="${ARCH:-}"
[ -n "${ARCH_EXPECTED_RAW}" ] || die "ARCH is required and must be set to 'cpu' or 'gpu'."
ARCH_EXPECTED="${ARCH_EXPECTED_RAW,,}"

case "${ARCH_EXPECTED}" in
  cpu|gpu) ;;
  *) die "Invalid ARCH='${ARCH_EXPECTED_RAW}'. Use 'cpu' or 'gpu'." ;;
esac

detect_runtime_arch() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "gpu"
  else
    echo "cpu"
  fi
}

ARCH_DETECTED="$(detect_runtime_arch)"

echo "=== Architecture ==="
echo "  expected_arch = ${ARCH_EXPECTED}"
echo "  detected_arch = ${ARCH_DETECTED}"

[ "${ARCH_DETECTED}" = "${ARCH_EXPECTED}" ] || \
  die "Architecture mismatch: expected '${ARCH_EXPECTED}', detected '${ARCH_DETECTED}'."

############################################################
# Run Julia simulation
############################################################

echo "=== Running Julia simulation (job: ${JOB_ID}, scenario: ${SCENARIO}) ==="
echo "Script: ${SCRIPT_PATH}"

julia --color=no --project=/app "${SCRIPT_PATH}" \
  --scenario "${SCENARIO}" \
  --arch "${ARCH_EXPECTED}" \
  --stop_days "${STOP_DAYS}" \
  "$@"

status=$?

if [ "${status}" -ne 0 ]; then
  echo "❌ Simulation exited with status ${status}"
else
  echo "✅ Simulation completed successfully."
fi

############################################################
# Upload output in cloud mode
############################################################

if [ -n "${GCS_BUCKET:-}" ]; then
  echo "=== Uploading output from ${LOCAL_OUTPUT_DIR} → gs://${GCS_BUCKET}/${GCS_OUTPUT_PREFIX}/${JOB_ID}/"
  # The wildcard must be outside quotes to expand, but we still quote the prefix.
  gsutil -m cp -r "${LOCAL_OUTPUT_DIR}/"* "gs://${GCS_BUCKET}/${GCS_OUTPUT_PREFIX}/${JOB_ID}/" || \
    echo "⚠️ No output files to upload."
else
  echo "✅ Local run complete. Output saved to: ${LOCAL_OUTPUT_DIR}"
fi

echo "=== Done (job: ${JOB_ID}) ==="
exit "${status}"
