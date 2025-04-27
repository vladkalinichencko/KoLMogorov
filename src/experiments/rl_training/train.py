"""
Script to train and evaluate a language model using REINFORCE to minimize program length.
"""
import argparse
import json
import logging
from pathlib import Path
import sys
from typing import List, Dict, Any
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from .environment import KolmogorovEnv
from .trainer import REINFORCETrainer
from ..evaluation.post_process_programs import post_process_programs
from ..evaluation.evaluate import evaluate

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrainingMetrics:
    """Class to track and plot training metrics."""
    def __init__(self):
        self.epochs = []
        self.rewards = []
        self.losses = []
        self.compression_rates = []
        self.accuracies = []
        
    def update(self, epoch: int, avg_reward: float, avg_loss: float):
        """Update training metrics for an epoch."""
        self.epochs.append(epoch)
        self.rewards.append(avg_reward)
        self.losses.append(avg_loss)
        
    def update_eval(self, compression_rate: float, accuracy: float):
        """Update evaluation metrics."""
        self.compression_rates.append(compression_rate)
        self.accuracies.append(accuracy)
        
    def plot(self, output_dir: Path):
        """Plot training and evaluation metrics."""
        # Create figure with multiple subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training metrics
        ax1.plot(self.epochs, self.rewards, 'b-')
        ax1.set_title('Average Reward per Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Reward')
        
        ax2.plot(self.epochs, self.losses, 'r-')
        ax2.set_title('Average Loss per Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        
        # Plot evaluation metrics
        epochs_eval = np.arange(len(self.compression_rates))
        ax3.plot(epochs_eval, self.compression_rates, 'g-')
        ax3.set_title('Compression Rate')
        ax3.set_xlabel('Evaluation')
        ax3.set_ylabel('Compression Rate')
        
        ax4.plot(epochs_eval, self.accuracies, 'm-')
        ax4.set_title('Program Accuracy')
        ax4.set_xlabel('Evaluation')
        ax4.set_ylabel('Accuracy (%)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_metrics.png')
        plt.close()

def load_sequences(data_path: str) -> List[List]:
    """Load sequences from a jsonl file, ignoring the example programs."""
    sequences = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            # Extract sequence from instruction prompt
            # Find the sequence between the first square brackets after "following sequence:"
            text = item["text"]
            seq_start = text.find("following sequence:") 
            if seq_start != -1:
                # Find the first [ after "following sequence:"
                start = text.find("[", seq_start)
                end = text.find("]", start) + 1
                if start != -1 and end != -1:
                    try:
                        # Clean and parse the sequence
                        sequence_str = text[start:end].strip()
                        sequence = eval(sequence_str)
                        sequences.append(sequence)
                    except Exception as e:
                        print(f"Error parsing sequence: {e}")
                        print(f"Problematic text: {text[start:end]}")
            else:
                print(f"Could not find sequence in: {text[:100]}...")
    return sequences
    
def evaluate_model(
    trainer: REINFORCETrainer,
    eval_data_path: str,
    output_dir: Path,
    epoch: int
) -> Dict[str, float]:
    """Evaluate model by generating programs and computing true compression metrics."""
    logging.info("Running evaluation...")
    
    eval_sequences = load_sequences(eval_data_path)
    results = []
    
    for seq in tqdm(eval_sequences, desc="Generating programs"):
        # Use the dedicated generate_program method
        program = trainer.generate_program(seq)
        
        results.append({
            "sequence": seq,
            "generation": f"The Python program that generates the sequence is:\n{program}\n###"
        })
    
    # Save generated programs
    eval_output = output_dir / f"generated_programs_epoch_{epoch}.jsonl"
    with open(eval_output, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
            
    # Run paper's evaluation pipeline
    post_processed_path = output_dir / f"post_processed_epoch_{epoch}.jsonl"
    evaluated_path = output_dir / f"evaluated_epoch_{epoch}.jsonl"
    
    post_process_programs(str(eval_output), str(post_processed_path))
    evaluate(str(post_processed_path), str(evaluated_path))
    
    # Read evaluation results
    df = pd.read_csv(str(evaluated_path).replace(".jsonl", ".csv"))
    compression_rate = float(df['compression_rate'].mean() if 'compression_rate' in df else float('inf'))
    accuracy = float(df['ex_acc'].mean() * 100 if 'ex_acc' in df else 0)
    
    return {
        "compression_rate": compression_rate,
        "accuracy": accuracy
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name/path of pretrained model to use")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data jsonl file")
    parser.add_argument("--eval_data_path", type=str, required=True, help="Path to evaluation data jsonl file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--reward_constant", type=float, default=100.0, help="Base reward for correct programs")
    parser.add_argument("--length_penalty", type=float, default=0.1, help="Penalty coefficient for program length")
    parser.add_argument("--checkpoint_freq", type=int, default=5, help="Save checkpoints every N epochs")
    parser.add_argument("--eval_freq", type=int, default=5, help="Run evaluation every N epochs")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize environment and trainer
    env = KolmogorovEnv(
        reward_constant=args.reward_constant,
        length_penalty=args.length_penalty
    )
    
    trainer = REINFORCETrainer(
        model_name=args.model_name,
        env=env,
        learning_rate=args.learning_rate
    )
    
    # Initialize metrics tracker
    metrics = TrainingMetrics()
    
    # Load training data
    sequences = load_sequences(args.data_path)
    best_compression_rate = float('inf')
    
    # Create run directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True)
    
    # Save config
    with open(run_dir / "config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
        
    # Setup logging to file
    file_handler = logging.FileHandler(run_dir / "train.log")
    logging.getLogger().addHandler(file_handler)
    
    # Training loop
    for epoch in range(args.num_epochs):
        total_reward = 0
        total_loss = 0
        num_episodes = 0
        
        # Process sequences in batches
        for i in tqdm(range(0, len(sequences), args.batch_size), desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            batch_sequences = sequences[i:i + args.batch_size]
            
            # Train on each sequence in batch
            for sequence in batch_sequences:
                metrics_step = trainer.train_step(sequence)
                total_reward += metrics_step["total_reward"]
                total_loss += metrics_step["policy_loss"]
                num_episodes += 1
                
        # Calculate epoch metrics
        avg_reward = total_reward / num_episodes
        avg_loss = total_loss / num_episodes
        logging.info(f"Epoch {epoch+1} - Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}")
        
        # Update training metrics
        metrics.update(epoch + 1, avg_reward, avg_loss)
        
        # Run evaluation
        if (epoch + 1) % args.eval_freq == 0:
            eval_metrics = evaluate_model(trainer, args.eval_data_path, run_dir, epoch + 1)
            metrics.update_eval(eval_metrics["compression_rate"], eval_metrics["accuracy"])
            
            logging.info(f"Evaluation - Compression Rate: {eval_metrics['compression_rate']:.4f}, " f"Accuracy: {eval_metrics['accuracy']:.2f}%")
            
            # Save if best model
            if eval_metrics["compression_rate"] < best_compression_rate:
                best_compression_rate = eval_metrics["compression_rate"]
                torch.save({
                    "model_state": trainer.model.state_dict(),
                    "optimizer_state": trainer.optimizer.state_dict(),
                    "epoch": epoch,
                    "metrics": eval_metrics,
                    "args": vars(args)
                }, run_dir / "best_model.pt")
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0:
            torch.save({
                "model_state": trainer.model.state_dict(),
                "optimizer_state": trainer.optimizer.state_dict(),
                "epoch": epoch,
                "metrics": {"avg_reward": avg_reward, "avg_loss": avg_loss},
                "args": vars(args)
            }, run_dir / f"checkpoint-{epoch+1}.pt")
            
        # Plot metrics
        metrics.plot(run_dir)
    
    # Final evaluation
    final_metrics = evaluate_model(trainer, args.eval_data_path, run_dir, args.num_epochs)
    logging.info("Training completed.")
    logging.info(f"Final Evaluation - Compression Rate: {final_metrics['compression_rate']:.4f}, " f"Accuracy: {final_metrics['accuracy']:.2f}%")

    # Save final model
    torch.save({
        "model_state": trainer.model.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "epoch": args.num_epochs,
        "metrics": final_metrics,
        "args": vars(args)
    }, run_dir / "final_model.pt")

if __name__ == "__main__":
    main()
