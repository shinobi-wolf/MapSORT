#!/usr/bin/env bash
# CPU-only COLMAP sparse reconstruction, tuned for scripts/landmarks/sfm_mapping.py.
# Usage: sparse_recon.sh <image_dir> <out_dir>
# Env: MAX_FEATURES (16384), MAX_SIZE (3840), MATCHER (exhaustive|sequential), THREADS (-1 = all),
#      CAMERA_PARAMS (RADIAL "f,cx,cy,k1,k2"; set it when images carry no EXIF), START (extract|match|map)
# Writes <out_dir>/{database.db, sparse/<k>/, refined/, model_txt/, 0*_*.log}; model_txt is the largest model.
set -euo pipefail

[[ $# -eq 2 ]] || { sed -n '2,6p' "$0"; exit 1; }
IMAGES=$(realpath "$1")
OUT=$(realpath -m "$2")
MAX_FEATURES=${MAX_FEATURES:-16384}
MAX_SIZE=${MAX_SIZE:-3840}
MATCHER=${MATCHER:-exhaustive}
THREADS=${THREADS:--1}
CAMERA_PARAMS=${CAMERA_PARAMS:-}
START=${START:-extract}
DB=$OUT/database.db

case $START in extract) stage=0 ;; match) stage=1 ;; map) stage=2 ;;
  *) echo "unknown START=$START" >&2; exit 1 ;; esac

mkdir -p "$OUT/sparse" "$OUT/model_txt"
cd "$OUT"

# Full resolution, no upsampling: COLMAP drops the finest octaves first when over the cap,
# and those carry the small objects (fruit, bunches) landmarks are read from.
if (( stage <= 0 )); then
  colmap feature_extractor \
    --database_path "$DB" --image_path "$IMAGES" \
    --ImageReader.camera_model RADIAL --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 0 --SiftExtraction.num_threads "$THREADS" \
    --SiftExtraction.max_image_size "$MAX_SIZE" \
    --SiftExtraction.first_octave 0 \
    --SiftExtraction.max_num_features "$MAX_FEATURES" \
    2>&1 | tee 01_extract.log
fi

# Exhaustive so revisits match their first pass and share 3D points.
# No guided matching: on CPU (3.9.1) it is brute force and ran out of memory (15 GB) on 8 images.
if (( stage <= 1 )); then
  MATCH_OPTS=(--database_path "$DB" --SiftMatching.use_gpu 0 --SiftMatching.num_threads "$THREADS")
  case $MATCHER in
    exhaustive) colmap exhaustive_matcher "${MATCH_OPTS[@]}" 2>&1 | tee 02_match.log ;;
    sequential) colmap sequential_matcher "${MATCH_OPTS[@]}" --SequentialMatching.overlap 20 \
                  2>&1 | tee 02_match.log ;;
    *) echo "unknown MATCHER=$MATCHER" >&2; exit 1 ;;
  esac
fi

# Without EXIF COLMAP guesses f = 1.2 * width; on GrapeMOTS that guess sent focal and distortion
# diverging and no model grew past 11 images. A known focal is held fixed while mapping.
refine=1
if [[ -n $CAMERA_PARAMS ]]; then
  python3 - "$DB" "$CAMERA_PARAMS" <<'EOF'
import array, sqlite3, sys
db, params = sys.argv[1], [float(v) for v in sys.argv[2].split(",")]
con = sqlite3.connect(db)
con.execute("UPDATE cameras SET params = ?, prior_focal_length = 1", (array.array("d", params).tobytes(),))
con.commit()
EOF
  refine=0
fi

colmap mapper \
  --database_path "$DB" --image_path "$IMAGES" --output_path "$OUT/sparse" \
  --Mapper.num_threads "$THREADS" \
  --Mapper.ba_refine_focal_length "$refine" --Mapper.ba_refine_extra_params "$refine" \
  2>&1 | tee 03_mapper.log

best=""; best_n=-1
for m in "$OUT"/sparse/*/; do
  n=$(colmap model_analyzer --path "$m" 2>&1 | grep -oP 'Registered images: \K\d+')
  echo "$m: $n images" | tee -a 04_models.log
  if (( n > best_n )); then best=$m; best_n=$n; fi
done
colmap model_analyzer --path "$best" 2>&1 | tee -a 04_models.log

# Once the full model is up, intrinsics are well constrained; refine them in one final BA.
if (( refine == 0 )); then
  mkdir -p "$OUT/refined"
  colmap bundle_adjuster --input_path "$best" --output_path "$OUT/refined" \
    --BundleAdjustment.refine_focal_length 1 --BundleAdjustment.refine_extra_params 1 \
    2>&1 | tee 05_refine.log
  colmap model_analyzer --path "$OUT/refined" 2>&1 | tee -a 05_refine.log
  best=$OUT/refined
fi

colmap model_converter --input_path "$best" --output_path "$OUT/model_txt" --output_type TXT
grep -v '^#' "$OUT/model_txt/cameras.txt" | tee -a 04_models.log
echo "model_txt <- $best ($best_n images)" | tee -a 04_models.log
