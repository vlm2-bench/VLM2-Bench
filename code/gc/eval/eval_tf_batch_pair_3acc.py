import os
import json
import re
import argparse
from collections import defaultdict

def load_jsonl(file_path):
    """Load a JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

def save_jsonl(data, file_path):
    """Save a list of dicts to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def analyze_correct_answers(input_folder, output_folder):
    """
    Analyze T/F outputs of each model in input_folder:
    1. Process all jsonl files in the folder
    2. Group by q_id after reading
    3. Calculate accuracy for positive questions, negative questions, and both correct
    4. Save valid entries to output_folder
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Create correct and wrong subdirectories
    correct_dir = os.path.join(output_folder, 'correct')
    wrong_dir = os.path.join(output_folder, 'wrong')
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(wrong_dir, exist_ok=True)

    # Iterate through all jsonl files in folder
    for file_name in os.listdir(input_folder):
        if not file_name.endswith('.jsonl'):
            continue

        file_path = os.path.join(input_folder, file_name)
        data = load_jsonl(file_path)

        # Group by q_id
        grouped_data = defaultdict(list)
        for item in data:
            grouped_data[item["q_id"]].append(item)

        # For statistics
        correct_p = 0  # Number of correct positive examples
        correct_n = 0  # Number of correct negative examples  
        correct_pairs = 0  # Number of pairs both correct
        total_pairs = len(grouped_data)

        # Output file paths
        model_name = os.path.splitext(file_name)[0]  # Use filename without extension
        correct_file = os.path.join(correct_dir, f'{model_name}_correct.jsonl')
        wrong_file = os.path.join(wrong_dir, f'{model_name}_wrong.jsonl')

        with open(correct_file, 'w', encoding='utf-8') as correct_f, \
             open(wrong_file, 'w', encoding='utf-8') as wrong_f:

            for omni_id, items in grouped_data.items():
                p_correct = False
                n_correct = False
                
                for item in items:
                    if item["gt_answer"] == "T" and item["model_answer"] == item["gt_answer"]:
                        p_correct = True
                    elif item["gt_answer"] == "F" and item["model_answer"] == item["gt_answer"]:
                        n_correct = True

                if p_correct:
                    correct_p += 1
                if n_correct:
                    correct_n += 1
                if p_correct and n_correct:
                    correct_pairs += 1
                    for item in items:
                        correct_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                else:
                    for item in items:
                        wrong_f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # Calculate three accuracy metrics
        positive_acc = correct_p / total_pairs if total_pairs > 0 else 0
        negative_acc = correct_n / total_pairs if total_pairs > 0 else 0
        total_acc = correct_pairs / total_pairs if total_pairs > 0 else 0

        # Print model's three accuracy metrics (two-line format)
        print(f"Model: {model_name}")
        print(f"Positive Accuracy: {positive_acc:.2%} ({correct_p}/{total_pairs}) | Negative Accuracy: {negative_acc:.2%} ({correct_n}/{total_pairs}) | Total Accuracy: {total_acc:.2%} ({correct_pairs}/{total_pairs})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze model answers for T/F questions.')
    parser.add_argument('--input_folder', type=str, required=True, help='Input folder containing model output files')
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = os.path.join(input_folder, "real_correct")

    analyze_correct_answers(input_folder, output_folder)
    print("Done.")

'''sample usage:
python code/gc/eval/eval_tf_batch_pair_3acc.py --input_folder "code/gc/test/test_res/test_mat"
'''
