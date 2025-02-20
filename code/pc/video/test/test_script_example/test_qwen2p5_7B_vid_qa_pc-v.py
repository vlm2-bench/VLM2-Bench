import os
import json
from datetime import datetime
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm  # Optional, for progress bar
import argparse  # Import argparse for command-line arguments

# Load the pretrained Qwen2-VL model
model_path = "/root/pretrain_weights/Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,   # Use bfloat16 for better performance
    attn_implementation="flash_attention_2",  # Enable FlashAttention 2
    device_map="auto"             # Automatically map model to available devices
).eval()                          # Set model to evaluation mode

# Initialize the processor
processor = AutoProcessor.from_pretrained(model_path)

# Function to process a single video and question
def process_video_and_question(vid, question, video_folder):
    """
    Process a single video and question pair to generate an answer.

    Args:
        vid (str): Video ID.
        question (str): User question about the video.
        video_folder (str): Path to the folder containing video files.

    Returns:
        str: Generated answer from the model.
    """
    # Check if the video exists in the folder
    video_path = None
    for ext in [".mp4", ".mov", ".avi"]:
        potential_path = os.path.join(video_folder, f"{vid}{ext}")
        if os.path.exists(potential_path):
            video_path = potential_path
            break

    if not video_path:
        raise FileNotFoundError(f"Video for {vid} not found in {video_folder}")

    # Construct the message structure, including the video and the text query
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video_path}",  # Specify the video path with the 'file://' prefix
                    "max_pixels": 360 * 420,         # Optional: Set a maximum pixel resolution for the video
                    "fps": 1,                      # Optional: Set the frames per second for video processing, fps = 1 by default
                },
                {"type": "text", "text": question},  # Add the text query as part of the message
            ],
        }
    ]

    # Prepare the text input by applying the chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process the vision-related input (extract image or video frames)
    image_inputs, video_inputs = process_vision_info(messages)

    # Format inputs for the model, including text, video, and optional padding
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",  # Return PyTorch tensors for further processing
    )
    inputs = inputs.to("cuda")  # Move inputs to the GPU for inference

    # Generate the model's response based on the inputs
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    # Remove the original input tokens from the output tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode the generated token IDs into human-readable text
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Return the final output text
    return output_text[0].strip()

# Function to process batch of videos and questions
def process_batch(jsonl_input_path, video_folder, output_folder):
    """
    Process a batch of videos and questions from a JSONL file.

    Args:
        jsonl_input_path (str): Path to the input JSONL file.
        video_folder (str): Path to the folder containing video files.
        output_folder (str): Path to save the output JSONL file.

    Returns:
        None
    """
    with open(jsonl_input_path, "r") as f:
        input_data = [json.loads(line.strip()) for line in f]

    # Generate output file name based on model name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_last_part = os.path.basename(model_path)
    output_filename = f"vid_{model_name_last_part}_{timestamp}.jsonl"
    output_path = os.path.join(output_folder, output_filename)

    # Open output file in append mode
    with open(output_path, 'a') as f:
        for entry in tqdm(input_data):
            vid = entry["vid"]
            question = entry["question"]
            gt_answer = entry.get("gt_answer", "")

            try:
                model_answer = process_video_and_question(vid, question, video_folder)
            except Exception as e:
                print(f"Error processing video {vid}: {e}")
                model_answer = f"Error processing video {vid}: {str(e)}"

            entry["model_answer"] = model_answer
            entry["gt_answer"] = gt_answer
            
            # Write result immediately after processing each video
            f.write(json.dumps(entry) + "\n")
            f.flush()  # Ensure the line is written to disk

    print(f"Results saved to {output_path}")

def main():
    # Default paths
    default_question_file = "jsonl/pc/vallina/pc_v_open-ended.jsonl"  # Path to your questions JSONL file
    default_video_folder = "data/pc/video/UNI"  # Path to your video folder
    default_output_dir = "code/pc/video/test/test_res/test_pc_v_open-ended"  # Path to your output directory
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process questions and generate answers.")
    parser.add_argument('--question_file', type=str, default=default_question_file, help="Path to the questions JSONL file.")
    parser.add_argument('--video_folder', type=str, default=default_video_folder, help="Path to the video folder.")
    parser.add_argument('--output_dir', type=str, default=default_output_dir, help="Path to the output directory.")
    
    args = parser.parse_args()

    # Use the provided or default paths
    jsonl_input_path = args.question_file
    video_folder = args.video_folder
    output_folder = args.output_dir

    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process the batch of questions
    process_batch(jsonl_input_path, video_folder, output_folder)

if __name__ == "__main__":
    main()


'''sample usage, cd to base folder then run:
python code/pc/video/test/test_script_example/test_qwen2p5_7B_vid_qa_pc-v.py \
--question_file "jsonl/pc/vallina/pc_v_open-ended.jsonl" \
--video_folder "data/pc/video/UNI" \
--output_dir "code/pc/video/test/test_res/test_pc_v_open-ended"
'''