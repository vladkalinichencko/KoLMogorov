# Reinforcement Learning Training for Kolmogorov Complexity

This module implements a reinforcement learning (RL) approach to training language models for program synthesis, specifically aimed at minimizing program length to approximate Kolmogorov complexity. By training models to generate minimal yet correct programs, we can improve compression rates beyond what's possible with standard language model prompting.

## Overview & Methodology

The implementation uses the REINFORCE algorithm (Williams, 1992) to train language models to generate minimal Python programs that reproduce input sequences. The approach works as follows:

1. **Line-by-Line Program Generation**: The model generates programs one line at a time, allowing for fine-grained reward attribution
2. **Immediate Program Execution**: Each generated program is executed to verify it produces the target sequence
3. **Length-Based Rewards**: Programs receive rewards based on correctness and penalized based on length
4. **Credit Assignment**: The REINFORCE algorithm propagates final rewards back through all generation decisions

### Key Components

- **`environment.py`**: Defines the `KolmogorovEnv` class which:
  - Executes programs in a safe sandbox
  - Verifies program output against target sequences
  - Calculates rewards based on program correctness and length
  
- **`trainer.py`**: Implements the REINFORCE algorithm with:
  - Policy gradient optimization for language models
  - Program generation with controlled prompting
  - Output processing to extract valid Python code
  
- **`train.py`**: Main training script that:
  - Handles data loading and batching
  - Manages training loop and evaluation
  - Tracks metrics and saves checkpoints

## Usage

To train a model using reinforcement learning, run:

```bash
python -m src.experiments.rl_training.train \
  --model_name "deepseek-ai/deepseek-coder-1.3b-instruct" \
  --data_path "src/data/seqcoder_training_data/10k.jsonl" \
  --eval_data_path "src/data/sub_sequences_length_128_1mb/text.jsonl" \
  --output_dir "checkpoints/rl_kolmogorov" \
  --num_epochs 10 \
  --batch_size 8 \
  --learning_rate 1e-5 \
  --reward_constant 100.0 \
  --length_penalty 0.1 \
  --checkpoint_freq 1 \
  --eval_freq 1
```

### Parameters Explained

- **`model_name`**: HuggingFace model to fine-tune (code-specialized models work best)
- **`data_path`**: Path to training data in JSONL format with sequences to compress
- **`eval_data_path`**: Path to evaluation data for periodic testing
- **`output_dir`**: Directory where checkpoints and metrics will be saved
- **`num_epochs`**: Number of training epochs
- **`batch_size`**: Batch size for training (smaller values use less memory)
- **`learning_rate`**: Learning rate for the optimizer
- **`reward_constant`**: Base reward for correct programs (C)
- **`length_penalty`**: Penalty coefficient for program length (λ)
- **`checkpoint_freq`**: Save model checkpoints every N epochs
- **`eval_freq`**: Run evaluation every N epochs

### Memory Optimization Tips

If you encounter memory issues (especially on M1/M2 Macs), try:
- Reduce batch_size to 1-4
- Add `--device "cpu"` to run on CPU rather than GPU 
- Add `--use_8bit_quantization` if you implement this option
- Process a subset of the data by using `head -n 1000 10k.jsonl > 1k.jsonl`

## Environment Details

The `KolmogorovEnv` implements a standard reinforcement learning environment with:

- **State**: Current program text (empty at the beginning)
- **Action**: Adding a line of Python code to the current program
- **Reward**: `reward_constant - length_penalty * encoded_length` if the program correctly produces the target sequence, otherwise 0
- **Done**: When the program successfully produces the target sequence or max steps are reached

The environment executes the program after each step to check if it produces the correct output sequence. It includes safety measures to prevent harmful code execution by running programs in a restricted context.

## REINFORCE Implementation

The trainer uses the REINFORCE algorithm (policy gradient) with these steps:

1. **Model Prompt**: A carefully crafted prompt instructs the model to generate code one line at a time using the specific DSL functions
2. **Action Sampling**: For each step, sample the next line of code from the model's probability distribution
3. **Environment Feedback**: Execute the updated program and get a reward (positive only when correct)
4. **Credit Assignment**: Calculate discounted returns to attribute the final reward to all previous decisions
5. **Policy Update**: Update the model to increase the probability of actions that led to high rewards

### Training Loop

For each sequence in the dataset:
1. Reset the environment with the target sequence
2. Generate a program line-by-line until success or max steps
3. Calculate discounted returns and policy gradient
4. Update model parameters to maximize expected reward

## Results and Metrics

The training process tracks several key metrics:

- **Compression Rate**: Average program length divided by sequence length
- **Program Accuracy**: Percentage of correctly generated programs
- **Average Reward**: Mean reward across all episodes
- **Policy Loss**: Loss value for the policy gradient updates

Training produces visualizations of these metrics over time to track progress.

## Code Structure and Extensions

The implementation is designed to be modular and extensible:

- **Multi-modal Training**: Works with any sequence data (numbers, text, DNA, audio features)
- **Model Agnostic**: Can use any HuggingFace auto-regressive language model
- **Environment Configuration**: Adjustable reward functions and execution parameters
- **Evaluation Pipeline**: Integrated with the paper's evaluation methodology
