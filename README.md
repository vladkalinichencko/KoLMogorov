# The KoLMogorov-Test

Official repository for the paper [*"The KoLMogorov Test: Compression by Code Generation"*](https://openreview.net/forum?id=C45YqeBDUM)

The Kolmogorov complexity of a sequence is the length of the shortest computer program that produces the sequence.

The aim of the KoLMogorov-Test (KT) is to empirically evaluate the ability of CodeLMs to detect patterns in and compress sequences by writing short programs that output them.

## 👋 Getting started

To get started, first [download the data](src/data).

KT currently includes six modalities - text, DNA, three encodings of audio data (MFCC, 16-bit, and 8-bit), and synthetic sequences produced by random programs. Two dataset sizes are available: a small one with 1MB per modality, and a large one with 1GB (DNA and text only).

For more information on making a submission, please check our [leaderboard](https://huggingface.co/spaces/KoLMogorov-Test/Leaderboard).

## 🧪 Experiments
Please check out my [experiments source code](src/experiments) to run experiments from the paper. Specifically, we provide information on how to run inference with prompted models and how to train our specialized SeqCoder models.

### Reinforcement Learning for Kolmogorov Complexity

I've implemented a new reinforcement learning approach to train language models for program synthesis that minimizes program length. This approach directly optimizes models to generate shorter, correct programs through policy gradient methods:

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
  --length_penalty 0.1
```

See [RL Training Documentation](src/experiments/rl_training/README.md) for details on the REINFORCE implementation, environment setup, and optimization techniques. 

## 📈 Evaluation
The main evaluation is simply the compression rate of the encoded programs and the relevant decoder. Please see our [leaderboard](https://huggingface.co/spaces/KoLMogorov-Test/Leaderboard) and our [evaluation source code](src/experiments/evaluation) for more information.

## 🧠 Synthetic data generation
In the paper, I show that for synthetic distributions where sampling program-sequence pairs is possible, I can train models with lower compression rate than current approaches. Please see our [synthetic data generation source code](src/synthetic_data_generation) for our DSL and data generation scripts.

## ✍️ Citation
```
@inproceedings{
    anonymous2024the,
    title={The Ko{LM}ogorov Test: Compression by Code Generation},
    author={Anonymous},
    booktitle={Submitted to The Thirteenth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=C45YqeBDUM},
    note={under review}
}
```

If you have any questions, please email us at [thekolmogorovtest@gmail.com](thekolmogorovtest@gmail.com)

## Licensing
The majority of code in this repository is licensed under CC-by-NC, however the third party code/files may be subject to different licenses.