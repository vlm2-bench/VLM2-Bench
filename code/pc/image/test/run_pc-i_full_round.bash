#!/bin/bash

# Initialize conda
source /root/miniconda3/etc/profile.d/conda.sh


# set your env and script
declare -A ENV_SCRIPTS=(
  ["qwen"]="code/pc/image/test/test_script_example/test_qwen2p5_7B_img_qa_pc.py" # ["conda env name"] = "path/to/your/script.py"
)


# Define three groups of file inputs
QUESTION_FILES=(
  "jsonl/pc/vanilla/pc_cpr.jsonl" # cpr
  "jsonl/pc/vanilla/pc_cnt.jsonl" # cnt
  "jsonl/pc/vanilla/pc_grp.jsonl" # grp
)


IMAGE_FOLDERS=(
  "data/pc/image/picks_face"
  "data/pc/image/picks_face"
  "data/pc/image/picks_face"
)


OUTPUT_DIRS=(
  "code/pc/image/test/test_res/test_cpr"  
  "code/pc/image/test/test_res/test_cnt"  
  "code/pc/image/test/test_res/test_grp"  
)

# Define a function to check and create directories
ensure_directory_exists() {
  local dir_path=$1
  if [ ! -d "$dir_path" ]; then
    echo "Output directory does not exist. Creating: $dir_path"
    mkdir -p "$dir_path"
  else
    echo "Output directory already exists: $dir_path"
  fi
}

# Check and create each output directory
for OUTPUT_DIR in "${OUTPUT_DIRS[@]}"; do
  ensure_directory_exists "$OUTPUT_DIR"
done


# Execute scripts
for ENV in "${!ENV_SCRIPTS[@]}"; do
  echo "Activating environment: $ENV"
  conda activate "$ENV"

  for i in ${!QUESTION_FILES[@]}; do
    echo "Running iteration $((i + 1)) with:"
    echo "  Question file: ${QUESTION_FILES[$i]}"
    echo "  Image folder: ${IMAGE_FOLDERS[$i]}"
    echo "  Output directory: ${OUTPUT_DIRS[$i]}"

    for SCRIPT in ${ENV_SCRIPTS[$ENV]}; do
      echo "Running script: $SCRIPT"
      python $SCRIPT \
        --question_file "${QUESTION_FILES[$i]}" \
        --image_folder "${IMAGE_FOLDERS[$i]}" \
        --output_dir "${OUTPUT_DIRS[$i]}" || {
        echo "Error running $SCRIPT"
        exit 1
      }
    done
  done

  echo "Deactivating environment: $ENV"
  conda deactivate
done
