#!/bin/bash

# Initialize conda
source /root/miniconda3/etc/profile.d/conda.sh


# set your env and script
declare -A ENV_SCRIPTS=(
  ["qwen"]="code/pc/video/test/test_script_example/test_qwen2p5_7B_vid_qa_pc-v.py" # ["conda env name"] = "path/to/your/script.py"
)


JSONL_INPUT_FILES=(
  "jsonl/pc/vallina/pc_v_open-ended.jsonl"
)


VIDEO_FOLDERS=(
  "data/pc/video/UNI"
)


OUTPUT_FOLDERS=(
  "code/pc/video/test/test_res/test_pc_v_open-ended"
)


# Check if the lengths of the input arrays are consistent
NUM_SETS=${#JSONL_INPUT_FILES[@]}
if [[ ${#VIDEO_FOLDERS[@]} -ne $NUM_SETS || ${#OUTPUT_FOLDERS[@]} -ne $NUM_SETS ]]; then
  echo "The number of input files, video folders, and output folders do not match. Please ensure they are equal."
  exit 1
fi

# Function to check and create directories if they do not exist
ensure_directory_exists() {
  local dir_path=$1
  if [ ! -d "$dir_path" ]; then
    echo "Output directory does not exist. Creating: $dir_path"
    mkdir -p "$dir_path" || {
      echo "Failed to create directory: $dir_path"
      exit 1
    }
  else
    echo "Output directory already exists: $dir_path"
  fi
}

# Check and create each output directory
for OUTPUT_DIR in "${OUTPUT_FOLDERS[@]}"; do
  ensure_directory_exists "$OUTPUT_DIR"
done

# Execute the scripts
for ENV in "${!ENV_SCRIPTS[@]}"; do
  echo "Activating environment: $ENV"
  conda activate "$ENV" || {
    echo "Failed to activate environment: $ENV"
    exit 1
  }

  for i in "${!JSONL_INPUT_FILES[@]}"; do
    echo "Running set $((i + 1)) with:"
    echo "  Input file: ${JSONL_INPUT_FILES[$i]}"
    echo "  Video folder: ${VIDEO_FOLDERS[$i]}"
    echo "  Output directory: ${OUTPUT_FOLDERS[$i]}"

    for SCRIPT in ${ENV_SCRIPTS[$ENV]}; do
      echo "Running script: $SCRIPT"
      python "$SCRIPT" \
        --jsonl_input_path "${JSONL_INPUT_FILES[$i]}" \
        --video_folder "${VIDEO_FOLDERS[$i]}" \
        --output_folder "${OUTPUT_FOLDERS[$i]}" || {
          echo "Error running script $SCRIPT"
          conda deactivate
          exit 1
        }
    done
  done

  echo "Deactivating environment: $ENV"
  conda deactivate || {
    echo "Failed to deactivate environment: $ENV"
    exit 1
  }
done

echo "All tasks have been completed successfully."
