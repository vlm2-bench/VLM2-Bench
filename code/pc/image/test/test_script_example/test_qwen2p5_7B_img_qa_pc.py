import os
import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datetime import datetime, timedelta
import time
from tqdm import tqdm  # Import tqdm for progress bar
import argparse  # Import argparse for command-line arguments

# Path to the pretrained model
model_path = "/root/pretrain_weights/Qwen/Qwen2.5-VL-7B-Instruct"

# Load the model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,   # Using bfloat16 for better performance
    attn_implementation="flash_attention_2",  # Enable FlashAttention2
    device_map="auto"             # Automatically map to GPU
).eval()

# Load the processor
processor = AutoProcessor.from_pretrained(model_path)

# Function to load and process images
def process_images_for_qwen(image_paths, question, processor):
    """
    Prepare the input messages and process images for Qwen model.
    """
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                # Dynamically generate image tokens for each image
                *[{"type": "image", "image": f"file://{path}"} for path in image_paths],
                {"type": "text", "text": question},
            ],
        }
    ]
    # Generate chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Process vision information
    image_inputs, video_inputs = process_vision_info(messages)
    # Create inputs for the model
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to("cuda")

# Function to process the batch
def process_batch(question_file, image_folder, output_dir, model_name):
    with open(question_file, 'r') as f:
        questions = [json.loads(line) for line in f]

    os.makedirs(output_dir, exist_ok=True)

    # Add timestamp to output file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{model_name}_{timestamp}_answers.jsonl")

    with open(output_file, 'w') as ans_file:
        total_questions = len(questions)
        start_time = time.time()  # Start time for elapsed time tracking

        # Use tqdm to create a progress bar for batch processing
        # pbar = tqdm(questions, desc="Processing Questions", unit="question")
        for idx, question_item in enumerate(tqdm(questions)):
            q_id = question_item["q_id"]
            obj_type = question_item["type"]
            question = question_item["question"]
            gt_answer = question_item["gt_answer"]

            '''concat image path to image base folder'''
            images = question_item["images"]
            image_paths = [os.path.join(image_folder, image) for image in images]

            if len(image_paths) < 2:
                print(f"Skipping {q_id} due to insufficient images.")
                continue

            system_prompt = (
                    ""
            )

            # Combine system prompt and question
            full_question = f"{system_prompt}\n{question}"

            # Process images and generate input tensor
            inputs = process_images_for_qwen(image_paths, full_question, processor)

            # Generate the model's response
            generated_ids = model.generate(**inputs, max_new_tokens=2048)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            ans_file.write(json.dumps({
                "q_id": q_id,
                "type": obj_type,
                "question": question,
                "model_answer": output_text[0].strip(),  # Extract the first response
                "gt_answer": gt_answer
            }) + "\n")
            ans_file.flush()
            print(f"Processed {q_id}: {output_text[0].strip()}")

def main():
    # Default paths
    default_question_file = "jsonl/pc/vanilla/pc_cpr.jsonl"  # Path to your questions JSONL file
    default_image_folder = "data/pc/image/picks_face"  # Path to your image folder
    default_output_dir = "code/pc/image/test/test_res/test_cpr"  # Path to your output directory
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process questions and generate answers.")
    parser.add_argument('--question_file', type=str, default=default_question_file, help="Path to the questions JSONL file.")
    parser.add_argument('--image_folder', type=str, default=default_image_folder, help="Path to the image folder.")
    parser.add_argument('--output_dir', type=str, default=default_output_dir, help="Path to the output directory.")
    
    args = parser.parse_args()

    # Use the provided or default paths
    question_file = args.question_file
    image_folder = args.image_folder
    output_dir = args.output_dir

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process the batch of questions
    process_batch(question_file, image_folder, output_dir, model_name="Qwen2.5-VL-7B-Instruct")

if __name__ == "__main__":
    main()


'''sample usage, cd to base folder then run:
python code/pc/image/test/test_script_example/test_qwen2p5_7B_img_qa_pc.py \
--question_file "jsonl/pc/vanilla/pc_cpr.jsonl" \
--image_folder "data/pc/image/picks_face" \
--output_dir "code/pc/image/test/test_res/test_cpr"
'''