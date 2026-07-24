#!/bin/bash
# Unpack downloaded DSEC archives into the layout dsec_dataset_lite expects, then verify.
#
#   ./setup_data.sh <zip_dir> <data_dir> [sequence ...]
#
# <zip_dir>   holds <seq>_events_left.zip, <seq>_optical_flow_forward_event.zip and
#             <seq>_optical_flow_forward_timestamps.txt[.txt]
# <data_dir>  becomes <data_dir>/train/<seq>/... i.e. pass the repo's data/dataset
#             (or the scratch dir that data/dataset symlinks to).
#
# With no sequences named, all 5 of the paper's validation split are unpacked.

set -euo pipefail

ZIP_DIR="${1:?usage: setup_data.sh <zip_dir> <data_dir> [sequence ...]}"
DATA_DIR="${2:?usage: setup_data.sh <zip_dir> <data_dir> [sequence ...]}"
shift 2

SEQUENCES=("$@")
if [ ${#SEQUENCES[@]} -eq 0 ]; then
    SEQUENCES=(thun_00_a zurich_city_02_d zurich_city_03_a zurich_city_08_a zurich_city_11_b)
fi

for seq in "${SEQUENCES[@]}"; do
    echo "=== ${seq} ==="
    events_dir="${DATA_DIR}/train/${seq}/events/left"
    flow_dir="${DATA_DIR}/train/${seq}/flow/forward"
    mkdir -p "${events_dir}" "${flow_dir}"

    # Both archives are flat, so they unzip straight into their target directory.
    unzip -o -q "${ZIP_DIR}/${seq}_events_left.zip" -d "${events_dir}"
    unzip -o -q "${ZIP_DIR}/${seq}_optical_flow_forward_event.zip" -d "${flow_dir}"

    # file_generator.py reads flow/forward_timestamps.txt, so drop the optical_flow_
    # prefix and any doubled .txt extension picked up during download.
    ts_src="${ZIP_DIR}/${seq}_optical_flow_forward_timestamps.txt.txt"
    [ -f "${ts_src}" ] || ts_src="${ZIP_DIR}/${seq}_optical_flow_forward_timestamps.txt"
    cp "${ts_src}" "${DATA_DIR}/train/${seq}/flow/forward_timestamps.txt"

    # A PNG/timestamp mismatch silently misaligns ground truth with events, because
    # _create_flow_maps numbers the sorted PNGs sequentiallyrather than by timestamp.
    n_png=$(find "${flow_dir}" -name '*.png' | wc -l)
    n_ts=$(grep -cv '^\s*\(#.*\)\?$' "${DATA_DIR}/train/${seq}/flow/forward_timestamps.txt")
    for f in events.h5 rectify_map.h5; do
        [ -f "${events_dir}/${f}" ] || { echo "  MISSING ${f}"; exit 1; }
    done
    if [ "${n_png}" -ne "${n_ts}" ]; then
        echo "  MISMATCH: ${n_png} PNGs vs ${n_ts} timestamp rows"
        exit 1
    fi
    echo "  OK: ${n_png} flow maps, events.h5 + rectify_map.h5 present"
done

echo
echo "Done. Preprocess with:  cd dsec_dataset_lite && python main.py ${SEQUENCES[*]}"
