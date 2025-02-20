<div align="center">

<img src="figs/vlm2-bench-icon_final.png" alt="icon" style=" height:95px;" />

# VLM²-Bench: A Closer Look at How Well VLMs Implicitly Link Explicit Matching Visual Cues

</div>

<div align="center">
<b>Jianshu Zhang<sup>1*</sup> Dongyu Yao<sup>2*</sup> Renjie Pi<sup>1</sup> Paul Pu Liang<sup>3</sup> Yi R. (May) Fung<sup>1</sup></b>

<sup>1</sup>HKUST &nbsp; <sup>2</sup>CMU &nbsp; <sup>3</sup>MIT

<sup>*</sup>: Equal contribution

[![arXiv](https://img.shields.io/badge/arXiv-2502.12084-B31B1B.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2502.12084)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-blue)](https://huggingface.co/datasets/Sterzhang/vlm2-bench)

</div>

---

## Abstract

We introduce VLM²-Bench, a benchmark designed to assess whether vision-language models (VLMs) can visually link matching cues. With 9 subtasks and over 3,000 test cases, VLM²-Bench evaluates models' abilities in recognition, OCR, knowledge, language generation, spatial awareness, and mathematical reasoning. Our comprehensive evaluation across eight open-source VLMs and the commercial GPT-4o reveals that even the best model lags behind human performance by approximately 34.80%. Our findings underscore critical challenges in visual feature extraction and cue integration, advocating for improved training paradigms to enhance independent reasoning and relational inference among visual cues.

---

## News

- **2025/02/18:** 🚀 We officially release VLM²-Bench—a benchmark focusing on the integrated capabilities of VLMs for linking explicit visual cues. This release features 9 subtasks with over 3,000 test cases.

---

## VLM$^2$-Bench Overview

VLM²-Bench is designed to evaluate models' ability to visually link matching cues across multiple images and videos. It is organized into three main categories:

- **General Cue (GC):** Assessing matching and tracking of visual elements.
- **Object-centric Cue (OC):** Evaluating comparison, counting, and grouping of objects.
- **Person-centric Cue (PC):** Focusing on comparing, counting, grouping, and video identity describing of individuals.

The dataset comprises over 3,000 question-answer pairs generated via a semi-automated pipeline with human verification, covering various question formats such as True/False, multiple-choice, numerical, and open-ended queries.

<div align="center">
<h4>VLM²-Bench Overview</h4>
<img src="figs/vlm2-bench_overview.png" width="80%" alt="VLM2-Bench Overview"/>
</div>

<br>

<div align="center">
<h4>Dataset Statistics</h4>
<img src="figs/vlm2-bench_statistics.png" width="80%" alt="VLM2-Bench Statistics"/>
</div>


---

## How to Evaluate Your Model on *VLM$^2$-Bench*


### Step 0: Environment Setup

- **Git clone VLM$^2$-Bench:**
```bash
git clone https://github.com/Sterzhang/VLM2-Bench.git
cd VLM2-Bench
```

- **Create a conda environment with Python 3.9:**
```bash
conda create -n vlm2bench python=3.9
conda activate vlm2bench
pip install openai>=1
pip install -r requirements.txt
```

For model inference, our benchmark does not require any specific packages. We recommend using the official inference scripts provided by model developers. For example, to test Qwen2.5-VL-7B-Instruct, you can follow the installation and inference instructions at [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).

### Step 1: Download the Data

- Download the VLM²-Bench dataset from our [huggingface repository](https://huggingface.co/datasets/Sterzhang/vlm2-bench/resolve/main/vlm2-bench_dataset.zip) link and unzip it at the root directory of this repository:

```bash
unzip vlm2-bench_dataset.zip
```

after unzip, you will see the following structure:

```bash
vlm2-bench/
├── code
│   ├── gc
│   ├── oc
│   ├── pc
├── data (images and videos)
│   ├── gc
│   ├── oc
│   ├── pc
├── jsonl (question files)
│   ├── gc
│   │   └── vanilla
│   │       └── gc_mat.jsonl
│   │       └── gc_trk.jsonl
│   ├── oc
│   ├── pc
```

### Step 2: Run Model Inference

- We provide example inference code for Qwen2.5-VL-7B under each task's test_script_example directory, for example: [code/gc/test/test_script_example/test_qwen2p5_7B_img_qa_gc.py](code/gc/test/test_script_example/test_qwen2p5_7B_img_qa_gc.py).

example usage for single model on gc_mat task:
```bash
python code/gc/test/test_script_example/test_qwen2p5_7B_img_qa_gc.py \
--question_file "jsonl/gc/vanilla/gc_mat.jsonl" \
--image_folder "data/gc/processed" \
--output_dir "code/gc/test/test_res/test_mat"
```


- Additionally, under the test directory of each task, there is a complete bash script for sequential testing on multiple models, for example: [code/gc/test/run_gc_full_round.bash](code/gc/test/run_gc_full_round.bash).

Example commands:

```bash
bash code/gc/test/run_gc_full_round.bash
```

this script will run the model for gc_mat and gc_trk tasks, and save the results in the `code/gc/test/test_res` directory.

For more details, please refer to the `.bash` scripts for each task directly. You may easily navigate to these files following the **Roadmap** below.

#### *Roadmap* of inference scripts and bash scripts for all tasks in VLM$^2$-Bench
example model: Qwen2.5-VL-7B-Instruct
- **GC**
  - inference script: [code/gc/test/test_script_example/test_qwen2p5_7B_img_qa_gc.py](code/gc/test/test_script_example/test_qwen2p5_7B_img_qa_gc.py)
  - bash script: [code/gc/test/run_gc_full_round.bash](code/gc/test/run_gc_full_round.bash)
- **OC**
  - inference script: [code/oc/test/test_script_example/test_qwen2p5_7B_img_qa_oc.py](code/oc/test/test_script_example/test_qwen2p5_7B_img_qa_oc.py)
  - bash script: [code/oc/test/run_oc_full_round.bash](code/oc/test/run_oc_full_round.bash)
- **PC-image**
  - inference script: [code/pc/image/test/test_script_example/test_qwen2p5_7B_img_qa_pc.py](code/pc/image/test/test_script_example/test_qwen2p5_7B_img_qa_pc.py)
  - bash script: [code/pc/image/test/run_pc-i_full_round.bash](code/pc/image/test/run_pc-i_full_round.bash)
- **PC-video** (open-ended)
  - inference script: [code/pc/video/test/test_script_example/test_qwen2p5_7B_vid_qa_pc-v.py](code/pc/video/test/test_script_example/test_qwen2p5_7B_vid_qa_pc-v.py)
  - bash script: [code/pc/video/test/run_pc-v_full_round.bash](code/pc/video/test/run_pc-v_full_round.bash)



### Step 3: Evaluate the Results

We provide separate evaluation scripts for each task as well as an all-in-one evaluation script (jupyter notebook) for evaluating all tasks.

- Navigate into the project directory, then run the evaluation script in `vlm2bench_evaluator.ipynb`. Remember to set the correct path to your result folder according to the instructions in the notebook.
- To evaluate the results of a single task, you can either run the script in notebook or run the bash script in the `eval` directory of the task (for example: [code/gc/eval/eval_tf_batch_pair_3acc.py](code/gc/eval/eval_tf_batch_pair_3acc.py)).

---

## Experimental Results
Leaderboard is shown below:

<div align="center">
<img src="figs/vlm2-bench_eval_results.png" width="80%" alt="Leaderboard"/>
</div>

Our evaluation on 8 state-of-the-art open-source vision-language models and GPT-4o shows:

- **Significant Performance Gap:** Even the best-performing model (GPT-4o) is on average ~34.80% behind human performance.
- **Diverse Performance Patterns:** Models exhibit distinct strengths and weaknesses across various visual cue categories, indicating the need for specialized improvements.


---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{zhang2025vlm2benchcloserlookvlms,
      title={VLM$^2$-Bench: A Closer Look at How Well VLMs Implicitly Link Explicit Matching Visual Cues}, 
      author={Jianshu Zhang and Dongyu Yao and Renjie Pi and Paul Pu Liang and Yi R. and Fung},
      year={2025},
      eprint={2502.12084},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.12084},
}
```

---

## License

- **Code:** Licensed under the [Apache 2.0 License](LICENSE).  
- **Dataset:** Licensed under the [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).

---

## Related Links
Other related research:
- [FuzzLLM](https://github.com/RainJamesY/FuzzLLM)
- [Image-Textualization](https://github.com/sterzhang/image-textualization)
- [PVIT Dataset](https://huggingface.co/datasets/Sterzhang/PVIT-3M)

