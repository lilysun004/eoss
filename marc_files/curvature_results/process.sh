#!/bin/bash
# Post-process curvature_results runs:
# 1. De-dupe by (optimizer, batch, params) — keep latest PCR folder.
# 2. Move npz + results.txt into each run's data/ subfolder.
# 3. Render plot_curvature_failure / plot_histograms / plot_tangent_drift (PNG only).
# 4. Move-rename all PNGs into curvature_results/plots/ as
#    <arch>__<run_folder_name>__<plot_type>.png.
#
# Idempotent: skips runs whose PNGs already exist in plots/. Re-run safely
# after any new array tasks finish.

set -uo pipefail

PY=/n/home06/mwalden/.conda/envs/eoss/bin/python
EOSS=/n/home06/mwalden/eoss
ROOT=$EOSS/marc_files/curvature_results
PLOTS=$ROOT/plots

mkdir -p "$PLOTS"

# Returns latest folder per (opt+batch+params) slug, filtered to those
# with both projections.npz and curvature_segment.npz (in root or data/).
latest_per_cell() {
    local dir="$1"
    cd "$dir" || return
    for d in */; do
        d="${d%/}"
        local has_p=0 has_c=0
        [ -f "$d/projections.npz" -o -f "$d/data/projections.npz" ] && has_p=1
        [ -f "$d/curvature_segment.npz" -o -f "$d/data/curvature_segment.npz" ] && has_c=1
        if [ $has_p -eq 1 ] && [ $has_c -eq 1 ]; then
            echo "$d"
        fi
    done | awk -F'_' '
        {
            ts = $1 "_" $2 "_" $3
            slug = substr($0, length(ts) + 2)
            if (!(slug in latest) || $0 > latest[slug]) latest[slug] = $0
        }
        END { for (s in latest) print latest[s] }
    '
}

# Move stray PNGs in run folder into PLOTS with arch__run__type.png naming.
move_pngs() {
    local arch="$1"
    local run="$2"  # absolute path to run folder
    local run_name=$(basename "$run")
    for png in "$run"/*.png; do
        [ -f "$png" ] || continue
        local base=$(basename "$png" .png)
        local type
        case "$base" in
            curvature_failure)     type="curvature_failure" ;;
            histograms_*_precond)  type="histograms_precond" ;;
            histograms_*)          type="histograms" ;;
            tangent_drift_*)       type="tangent_drift" ;;
            *)                     type="$base" ;;
        esac
        mv "$png" "$PLOTS/${arch}__${run_name}__${type}.png"
    done
}

process_folder() {
    local f="$1"     # absolute path to run folder
    local arch="$2"
    local run_name=$(basename "$f")

    # Tidy: move data files into data/.
    mkdir -p "$f/data"
    for n in projections.npz curvature_segment.npz distributions.npz results.txt; do
        if [ -f "$f/$n" ] && [ ! -f "$f/data/$n" ]; then
            mv "$f/$n" "$f/data/$n"
        elif [ -f "$f/$n" ] && [ -f "$f/data/$n" ]; then
            rm "$f/$n"
        fi
    done
    # Delete any straggler PDFs.
    find "$f" -maxdepth 1 -name '*.pdf' -delete 2>/dev/null

    # Skip if all 3 expected PNGs already in plots/ for this run.
    local curv="$PLOTS/${arch}__${run_name}__curvature_failure.png"
    local hist="$PLOTS/${arch}__${run_name}__histograms.png"
    local drift="$PLOTS/${arch}__${run_name}__tangent_drift.png"
    if [ -f "$curv" ] && [ -f "$hist" ] && [ -f "$drift" ]; then
        echo "  SKIP (already plotted): $run_name"
        return 0
    fi

    echo "  PLOT: $run_name"
    $PY $ROOT/plot_curvature_failure.py "$f" 2>&1 | tail -1 | sed 's/^/    [curv] /'
    $PY $EOSS/plot_histograms.py "$f"             2>&1 | tail -1 | sed 's/^/    [hist] /'
    $PY $EOSS/marc_files/drift_results/plot_tangent_drift.py "$f" 2>&1 | tail -1 | sed 's/^/    [drift] /'

    move_pngs "$arch" "$f"
}

for arch in mlp cnn vit sst; do
    arch_dir=$ROOT/curvature_failure_$arch
    [ -d "$arch_dir" ] || continue
    echo "=== $arch ==="
    for sub in $(latest_per_cell "$arch_dir"); do
        process_folder "$arch_dir/$sub" "$arch"
    done
done

echo ""
echo "Total PNGs in plots/: $(ls $PLOTS/*.png 2>/dev/null | wc -l)"
