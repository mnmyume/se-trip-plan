# Architectural Choices in AI Applications: A Software Engineering Approach to Performance Evaluation

This work aims to systematically evaluate performance differences across AI software architectures, providing insights into how workflow design choices influence accuracy, computational cost, and other key metrics.
Using a trip-planning scenario as a case study, this work adopts [TravelPlanner](https://osu-nlp-group.github.io/TravelPlanner/) as the benchmark system.

## Setup Environment

1. Create a conda environment and install dependencies:
```bash
conda create -n travelplanner python=3.9
conda activate travelplanner
pip install -r requirements.txt
```

2. Download the [database](https://drive.google.com/file/d/1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE/view?usp=drive_link) and unzip it to the `TravelPlanner` directory (i.e., `your/path/TravelPlanner`).

## Running

### Sole-Planning Mode

Run this step to get trip plan outputs.

```bash
export OUTPUT_DIR=../../outputs/output
export MODEL_NAME=gemma-3-27b-it
export SET_TYPE=validation
export STRATEGY=direct
# 5 different architectures are provide, which can be change by NODE_MODE: separate, merge_attra_accom, merge_attra_resta, merge_accom_resta, merge_all
export NODE_MODE=separate

cd tools/planner
python sole_planning.py  --set_type $SET_TYPE --output_dir $OUTPUT_DIR --model_name $MODEL_NAME --strategy $STRATEGY --node_mode $NODE_MODE
```

## Postprocess

Parse natural language plans into json formats. 

```bash
export MODEL_NAME=gemma-3-27b-it
export OUTPUT_DIR=outputs/output
export SET_TYPE=validation
export STRATEGY=direct
export MODE=sole-planning
export TMP_DIR=outputs/tmp
export SUBMISSION_DIR=outputs/submission
export NODE_MODE=separate


python -m postprocess.parsing --set_type $SET_TYPE --output_dir $OUTPUT_DIR --model_name $MODEL_NAME --strategy $STRATEGY --mode $MODE --tmp_dir $TMP_DIR --node_mode $NODE_MODE

python -m postprocess.element_extraction --set_type $SET_TYPE --output_dir $OUTPUT_DIR --model_name $MODEL_NAME --strategy $STRATEGY --mode $MODE --tmp_dir $TMP_DIR --node_mode $NODE_MODE

python -m postprocess.combination --set_type $SET_TYPE --output_dir $OUTPUT_DIR --model_name $MODEL_NAME --strategy $STRATEGY --mode $MODE  --submission_file_dir $SUBMISSION_DIR --node_mode $NODE_MODE
```

## Evaluation

For the TravelPlanner benchmark, only the validation set is available for offline evaluation.
For test set evaluation, please use the official [leaderboard](https://huggingface.co/spaces/osunlp/TravelPlannerLeaderboard).

```bash
export SET_TYPE=validation
export EVALUATION_FILE_PATH=../outputs/submission/validation/merge_all/validation_gemma-3-27b-it_direct_sole-planning_submission.jsonl

python -m evaluation.eval --set_type $SET_TYPE --evaluation_file_path $EVALUATION_FILE_PATH
```
