import json
import argparse
import re
import os
from openai import OpenAI

# Initialize API client
client = OpenAI(api_key="") # add your openai api key here


# Define two prompts
prompt_ab = """#Task
You are evaluating a model's ability to accurately distinguish between two different individuals, A and B, who appear sequentially in a video (first A, then B). Given a description, your task is to determine if the model explicitly identifies that the first person (A) and the second person (B) are different individuals.
#Return Format
You only need return a number after "Score:". If you think the model correctly identifies that the two appearances belong to different individuals, return "Score: 1". If you think the model fails to explicitly state that there are two different individuals, return "Score: 0".
#Description
{description}
"""

prompt_aba = """#Task
You are evaluating a model's ability to accurately distinguish between two different individuals, A and B, who appear sequentially in a video following an ABA pattern (first A, then B, then A again). Given a description, your task is to determine whether the model explicitly identifies that: (1) A and B are different individuals, and (2) The person in the final scene is the same as the first (A).
#Return Format
You only need return a number after "Score:". (1) If the model correctly describes that the video follows an ABA sequence, explicitly recognizing that the first and last appearances belong to the same person (A), while the middle appearance is a different person (B), return "Score: 2".
(2) If the model correctly identifies that there are two different people in the video (A and B) but does not explicitly mention that the last scene returns to A, return "Score: 1".
(3) If the model fails to recognize that two different individuals appear (e.g., treats all appearances as the same person or does not distinguish between A and B), return "Score: 0".
#Description
{description}
"""

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MLLM responses on video descriptions.")
    parser.add_argument('input_dir', type=str, help="Directory containing model answer JSONL files")
    return parser.parse_args()

def get_model_name(filename):
    # Extract model name from filename pattern like vid_InternVL2.5-26B_20250206_035327.jsonl
    match = re.match(r'(vid_[^_]+)', filename)
    if match:
        return match.group(1)
    return None

def compute_average_scores(output_path):
    ab_scores = []
    aba_scores = []
    
    # Read the output JSONL and extract scores
    with open(output_path, 'r') as outfile:
        for line in outfile:
            data = json.loads(line)
            score = data.get("score")
            score = re.search(r"Score:\s*(\d+)", score)
            score = int(score.group(1)) if score else None
            type_ = data.get("type")

            if score is not None:
                if type_ == "AB":
                    ab_scores.append(score)
                elif type_ == "ABA":
                    aba_scores.append(score)

    # Compute average scores
    avg_ab = sum(ab_scores) / len(ab_scores) if ab_scores else 0
    avg_aba = sum(aba_scores) / len(aba_scores) if aba_scores else 0

    mapped_avg_ab = (avg_ab / 1) * 100
    mapped_avg_aba = (avg_aba / 2) * 100
    avg = (mapped_avg_ab + mapped_avg_aba) / 2

    return {
        "model": os.path.basename(output_path),
        "ab_score": mapped_avg_ab,
        "aba_score": mapped_avg_aba,
        "avg_score": avg
    }

def process_file(input_file, scored_dir):
    model_name = get_model_name(os.path.basename(input_file))
    if not model_name:
        print(f"Could not extract model name from {input_file}")
        return None
        
    output_file = os.path.join(scored_dir, f"{model_name}_scored.jsonl")
    
    # Open input and output files
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Iterate through each line (JSON object)
        for line in infile:
            data = json.loads(line)
            vid = data["vid"]
            desc = data["model_answer"]
            
            # Select appropriate prompt
            if "ABA" in vid:
                prompt = prompt_aba
                type_ = "ABA"
            else:
                prompt = prompt_ab
                type_ = "AB"
                
            prompt_ = prompt.format(description=desc)
            
            # Call OpenAI API to get GPT response
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": prompt_
                    }
                ],
                max_tokens=300,
            )
            
            # Get GPT response
            score = response.choices[0].message.content
            print(f"Processed {vid}: {score}")
            
            # Write results to output file
            result = {
                "vid": vid,
                "type": type_,
                "score": score,
                "model_answer": desc,
            }
            outfile.write(json.dumps(result) + "\n")
            
    return output_file

def main():

    # Set input directory for model answers
    input_dir = "code/pc/video/test/test_res/test_pc_v_open-ended"
    
    # Create scored_files directory if it doesn't exist
    scored_dir = os.path.join(input_dir, "scored_files")
    os.makedirs(scored_dir, exist_ok=True)
    
    # Process all files in input directory
    results = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.jsonl'):
            input_file = os.path.join(input_dir, filename)
            print(f"\nProcessing {filename}...")
            
            output_file = process_file(input_file, scored_dir)
            if output_file:
                results.append(compute_average_scores(output_file))
    
    # Print batch results
    print("\nBatch Evaluation Results:")
    print("-" * 80)
    print(f"{'Model':<30} {'AB Score':<15} {'ABA Score':<15} {'Average':<15}")
    print("-" * 80)
    for result in results:
        print(f"{result['model']:<30} {result['ab_score']:.2f}%{' '*10} {result['aba_score']:.2f}%{' '*10} {result['avg_score']:.2f}%")

if __name__ == "__main__":
    main()
