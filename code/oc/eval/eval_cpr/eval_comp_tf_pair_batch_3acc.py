import json
import os
import re
from collections import defaultdict
import argparse

def parse_tf_answer(model_answer):
    """
    Extract 'T' or 'F' from the tf type model_answer.
    Supports various formats such as 'T', 'F', 'True', 'False', 't', 'f', and sentences containing these words.
    If multiple 'T'/'F' are found, return None and mark as multiple answers.
    If no 'T'/'F' is found, return None and mark as no answer found.
    """
    # Define matching pattern to match 't', 'f', 'true', 'false'
    pattern = re.compile(r'\b(t|f|true|false)\b', re.IGNORECASE)
    matches = pattern.findall(model_answer)

    # Extract all matched answers
    extracted = [match.upper()[0] for match in matches]  # 'true' -> 'T', 'false' -> 'F'

    if len(extracted) == 1:
        return extracted[0], None  # Return the extracted single answer, no error
    elif len(extracted) > 1:
        return None, 'multiple_answers_found'  # Multiple answers found
    else:
        return None, 'no_answer_found'  # No answers found

def load_model_answers(model_answer_file):
    """
    Load the model answer file, grouping answers by main id, 
    where each main id contains answers for '_p' and '_n'.
    Returns a dictionary where the key is the main id and the value is a sub-dictionary containing 'p' and 'n'.
    """
    model_answers_dict = defaultdict(dict)
    with open(model_answer_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            q_id = data.get('q_id')
            if not q_id or '_' not in q_id:
                continue
            main_id, suffix = q_id.split('_', 1)
            if suffix not in ['p', 'n']:
                continue
            model_answers_dict[main_id][suffix] = {
                'model_answer': data.get('model_answer', '').strip(),
                'gt_answer': data.get('gt_answer', '').strip().upper()
            }
    # print(f"Loaded {len(model_answers_dict)} main_ids with their answer pairs from {model_answer_file}.")
    return model_answers_dict

def evaluate_pair_correctness(model_answers_dict):
    """
    Evaluate the correctness of the model's main id answers.
    Count the correctness of positive (p), negative (n), and overall (both correct).
    """
    correct_p = 0  # Number of correct positive answers
    correct_n = 0  # Number of correct negative answers
    correct_pairs = 0  # Number of both correct answers
    total_pairs = 0  # Total pairs

    for main_id, suffix_dict in model_answers_dict.items():
        # Ensure both sub-questions exist
        if 'p' not in suffix_dict or 'n' not in suffix_dict:
            continue  # Skip incomplete pairs

        total_pairs += 1

        # Get answers for p and n
        model_answer_p = suffix_dict['p']['model_answer']
        gt_answer_p = suffix_dict['p']['gt_answer']

        model_answer_n = suffix_dict['n']['model_answer']
        gt_answer_n = suffix_dict['n']['gt_answer']

        # Parse model answers
        parsed_p, error_p = parse_tf_answer(model_answer_p)
        parsed_n, error_n = parse_tf_answer(model_answer_n)

        # Check if positive answer is correct
        is_correct_p = (parsed_p == gt_answer_p) if parsed_p else False
        if is_correct_p:
            correct_p += 1

        # Check if negative answer is correct
        is_correct_n = (parsed_n == gt_answer_n) if parsed_n else False
        if is_correct_n:
            correct_n += 1

        # Check if both are correct
        if is_correct_p and is_correct_n:
            correct_pairs += 1

    return correct_p, correct_n, correct_pairs, total_pairs

def process_model_file(model_file_path):
    """
    Process a single model answer file and evaluate its accuracy.
    """
    model_name = os.path.splitext(os.path.basename(model_file_path))[0]

    # Load all answers from the model
    model_answers_dict = load_model_answers(model_file_path)

    # Evaluate the model's answers
    correct_p, correct_n, correct_pairs, total = evaluate_pair_correctness(model_answers_dict)
    
    # Calculate three types of accuracy
    positive_acc = correct_p / total if total > 0 else 0
    negative_acc = correct_n / total if total > 0 else 0
    total_acc = correct_pairs / total if total > 0 else 0
    
    print(f"Model: {model_name}")
    print(f"Positive Accuracy: {positive_acc:.2%} ({correct_p}/{total}) | Negative Accuracy: {negative_acc:.2%} ({correct_n}/{total}) | Total Accuracy: {total_acc:.2%} ({correct_pairs}/{total})")

def process_result_folder(result_folder):
    """
    Process all model answer files in the result folder and perform pair cross-validation.
    """
    # Iterate through each model answer file in the result folder
    for filename in os.listdir(result_folder):
        if not filename.endswith(".jsonl"):
            continue  # Only process .jsonl files

        file_path = os.path.join(result_folder, filename)
        process_model_file(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate comparison task results.')
    parser.add_argument('--input_folder', type=str, required=True, help='Input folder containing model output files')
    args = parser.parse_args()

    # Check if the input folder exists
    if not os.path.isdir(args.input_folder):
        print(f"Result folder does not exist: {args.input_folder}")
    else:
        process_result_folder(args.input_folder)

'''sample usage:
python code/oc/eval/eval_cpr/eval_comp_tf_pair_batch_3acc.py --input_folder "code/oc/test/test_res/test_cpr"
'''
