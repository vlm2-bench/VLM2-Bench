import json
import os
import re
import math
import argparse

# Mapping from English number words to integers
NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
}

# Define penalty factor for inverse power exponent calculation (recommended value 1-3; higher value means more significant penalty for errors)
PENALTY_FACTOR = 10
# Maximum image sequence length (for difficulty weighting), fixed at 4 for this task
L_MAX = 4

def words_to_num(s):
    """
    Convert English number words to integers.
    Supports formats like 'twenty one', 'one hundred', 'one hundred and five' etc.
    """
    s = s.lower().replace('-', ' ').replace('and', ' ')
    tokens = s.split()
    total = 0
    current = 0
    for token in tokens:
        if token in NUM_WORDS:
            scale = NUM_WORDS[token]
            if scale in (100, 1000):
                if current == 0:
                    current = 1
                current *= scale
                total += current
                current = 0
            else:
                current += scale
        else:
            return None
    total += current
    return total if total != 0 else None

def extract_numbers(text):
    """
    Extract all numbers from text, whether Arabic numerals or English number words.
    Returns a list of integers.
    """
    text = text.lower()
    # Extract Arabic numerals
    digit_numbers = re.findall(r'\d+', text)
    digit_numbers = [int(num) for num in digit_numbers]
    # Extract English number words
    word_numbers = []
    pattern = re.compile(
        r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|'
        r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|'
        r'seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|'
        r'sixty|seventy|eighty|ninety|hundred|thousand)\b',
        re.IGNORECASE)
    matches = pattern.findall(text)
    if matches:
        words = []
        for match in matches:
            words.append(match)
        word_phrase = ' '.join(words)
        num = words_to_num(word_phrase)
        if num is not None:
            word_numbers.append(num)
    return digit_numbers + word_numbers

def parse_model_answer(model_answer):
    """
    Extract numbers from model_answer and convert to integers.
    Returns the number if exactly one number is found, otherwise returns None.
    """
    numbers = extract_numbers(model_answer)
    if len(numbers) == 1:
        return numbers[0]
    else:
        return None

def load_curated_questions(curated_file):
    """
    Load question information from original question file,
    with q_id as key and image sequence length as value:
      - Prioritize using "image_seq_len" field
      - If not present, check length of "image_seq" list
      - Otherwise default to image_len = 2
    """
    curated = {}
    with open(curated_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            q_id = data.get('q_id')
            if q_id is not None:
                if "image_seq_len" in data:
                    curated[q_id] = data["image_seq_len"]
                elif "image_seq" in data and isinstance(data["image_seq"], list):
                    curated[q_id] = len(data["image_seq"])
                else:
                    curated[q_id] = 2  # Default value
    return curated

def evaluate_model_response(jsonl_file, curated_questions):
    """
    Evaluate a single jsonl file:
      - Get image_len (image sequence length) from curated_questions using q_id
      - Calculate absolute error raw_diff = |model_answer - gt_answer| for each record
      - If answer is completely correct, normalized score is 100;
        otherwise use new calculation method:
          1. Calculate max_error = max(gt_answer - 1, image_len - gt_answer)
          2. Calculate relative_error = raw_diff / max_error
          3. Calculate difficulty weight: weight = L_MAX / image_len
          4. Use inverse power exponent to amplify error:
             penalty = weight * (relative_error ** (1.0 / PENALTY_FACTOR))
          5. Final normalized score = 100 * (1 - penalty) (score is 0 when penalty >= 1)
      - Accuracy calculation remains unchanged.
    """
    base_dir = os.path.dirname(jsonl_file)
    model_name = os.path.splitext(os.path.basename(jsonl_file))[0]
    
    correct_dir = os.path.join(base_dir, 'correct')
    wrong_dir   = os.path.join(base_dir, 'wrong')
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(wrong_dir, exist_ok=True)

    correct_file = os.path.join(correct_dir, f'{model_name}_correct.jsonl')
    wrong_file   = os.path.join(wrong_dir, f'{model_name}_wrong.jsonl')

    total_count = 0
    correct_count = 0
    valid_count = 0
    total_norm_score = 0

    with open(correct_file, 'w', encoding='utf-8') as correct_f, \
         open(wrong_file, 'w', encoding='utf-8') as wrong_f, \
         open(jsonl_file, 'r', encoding='utf-8') as file:

        for line in file:
            data = json.loads(line)
            total_count += 1

            model_answer = data.get('model_answer', '').strip()
            gt_answer = data.get('gt_answer', None)
            if gt_answer is None:
                data['error_reason'] = 'missing_gt_answer'
                wrong_f.write(json.dumps(data, ensure_ascii=False) + '\n')
                continue

            # Get image_len based on q_id
            q_id = data.get('q_id', 'unknown_q_id')
            if q_id in curated_questions:
                image_len = curated_questions[q_id]
            else:
                image_len = 2
                data['warning'] = 'q_id not found in curated questions, defaulting image_len=2'
            
            parsed_answer = parse_model_answer(model_answer)
            if parsed_answer is None:
                data['raw_diff'] = None
                data['normalized_score'] = 0.0
                data['error_reason'] = 'invalid_answer_format'
                wrong_f.write(json.dumps(data, ensure_ascii=False) + '\n')
                continue

            if not (1 <= parsed_answer <= image_len):
                raw_diff = abs(parsed_answer - gt_answer)
                data['raw_diff'] = raw_diff
                data['normalized_score'] = 0.0
                data['error_reason'] = 'answer_out_of_expected_range'
                wrong_f.write(json.dumps(data, ensure_ascii=False) + '\n')
                continue

            raw_diff = abs(parsed_answer - gt_answer)
            if raw_diff == 0:
                norm_score = 100.0
            else:
                # Calculate maximum possible error: consider gt_answer position
                max_error = max(gt_answer - 1, image_len - gt_answer)
                relative_error = raw_diff / max_error if max_error > 0 else 0
                # Difficulty weight: fewer images means higher weight
                weight = L_MAX / image_len
                # Use inverse power exponent to amplify error
                penalty = weight * (relative_error ** (1.0 / PENALTY_FACTOR))
                norm_score = 100 * (1 - penalty) if penalty < 1 else 0.0
            data['raw_diff'] = raw_diff
            data['normalized_score'] = norm_score

            total_norm_score += norm_score
            valid_count += 1

            if parsed_answer == gt_answer:
                correct_count += 1
                correct_f.write(json.dumps(data, ensure_ascii=False) + '\n')
            else:
                data['error_reason'] = 'incorrect_answer'
                wrong_f.write(json.dumps(data, ensure_ascii=False) + '\n')

    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    avg_norm_score = (total_norm_score / valid_count) if valid_count > 0 else 0

    print(f"Model: {model_name} Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
    print(f"Model: {model_name} Average Normalized Score: {avg_norm_score:.2f}")

def process_folder(input_folder, curated_questions):
    """Process all .jsonl files in the folder"""
    for filename in os.listdir(input_folder):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(input_folder, filename)
            evaluate_model_response(file_path, curated_questions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate counting task results.')
    parser.add_argument('--input_folder', type=str, required=True, help='Input folder containing model output files')
    args = parser.parse_args()

    # Original counting questions file path 
    curated_questions_file = 'jsonl/pc/vanilla/pc_cnt.jsonl' # path to pc cnt questions
    curated_questions = load_curated_questions(curated_questions_file)
    
    # Check if the input folder exists
    if not os.path.isdir(args.input_folder):
        print(f"Result folder does not exist: {args.input_folder}")
    else:
        process_folder(args.input_folder, curated_questions)

'''sample usage:
python code/pc/image/eval/eval_cnt/acc_with_cnt_score.py --input_folder "code/pc/image/test/test_res/test_cnt"
'''
