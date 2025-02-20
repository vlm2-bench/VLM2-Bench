import json
import os
import re
import argparse
def evaluate_model_response(jsonl_file):
    # Create two folders if they don't exist
    base_dir = os.path.dirname(jsonl_file)
    model_name = os.path.splitext(os.path.basename(jsonl_file))[0]  # Parse model name
    
    correct_dir = os.path.join(base_dir, 'correct')
    wrong_dir = os.path.join(base_dir, 'wrong')
    
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(wrong_dir, exist_ok=True)

    # Output file paths
    correct_file = os.path.join(correct_dir, f'{model_name}_correct.jsonl')
    wrong_file = os.path.join(wrong_dir, f'{model_name}_wrong.jsonl')

    # Open files in write mode, overwrite previous content
    correct_f = open(correct_file, 'w')
    wrong_f = open(wrong_file, 'w')

    def clean_answer(answer):
        """Remove the option letter and its following content, returning only the letter part."""
        return answer.split(')')[0].strip()

    def count_options(answer):
        """Count the number of options in the answer."""
        return len(re.findall(r'\([A-Z]\)', answer))

    correct_count = 0
    total_count = 0

    # Read file and process
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            model_answer = data['model_answer']
            gt_answer = data['gt_answer']
            case_id = data.get('id', 'unknown_id')  # Default to 'unknown_id' if 'id' is missing

            total_count += 1  # Count total entries

            # Handle multiple choice answers
            if count_options(model_answer) > 1:
                data['error_reason'] = 'multi-choice'
                wrong_f.write(json.dumps(data) + '\n')  # Classify multi-choice as wrong
                continue

            # Clean answers for comparison
            model_cleaned = clean_answer(model_answer)
            gt_cleaned = clean_answer(gt_answer)

            # Classify and process
            if model_cleaned == gt_cleaned:
                correct_f.write(json.dumps(data) + '\n')
                correct_count += 1  # Count correct answers
            else:
                data['error_reason'] = 'incorrect_answer'
                wrong_f.write(json.dumps(data) + '\n')

    # Print model accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Model: {model_name} Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")

    # Close file handles
    correct_f.close()
    wrong_f.close()

def process_folder(input_folder):
    """Batch process all .jsonl files in the folder"""
    for filename in os.listdir(input_folder):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(input_folder, filename)
            evaluate_model_response(file_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate grouping task results.')
    parser.add_argument('--input_folder', type=str, required=True, help='Input folder containing model output files')
    args = parser.parse_args()

    # Check if the input folder exists
    if not os.path.isdir(args.input_folder):
        print(f"Result folder does not exist: {args.input_folder}")
    else:
        process_folder(args.input_folder)

'''sample usage:
python code/pc/image/eval/eval_grp/eval_acc_batch_classify_mcq.py --input_folder "code/pc/image/test/test_res/test_grp"
'''
