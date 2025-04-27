"""
REINFORCE algorithm implementation for training language models to minimize program length.
"""
from typing import List, Dict, Any
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from .environment import KolmogorovEnv

class REINFORCETrainer:
    def __init__(
        self,
        model_name: str,
        env: KolmogorovEnv,
        learning_rate: float = 1e-5,
        gamma: float = 0.99,
        max_episode_steps: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the REINFORCE trainer.
        
        Args:
            model_name: Name/path of pretrained model to use
            env: KolmogorovEnv instance
            learning_rate: Learning rate for policy optimization
            gamma: Discount factor for rewards
            max_episode_steps: Maximum steps per episode
            device: Device to run model on
        """
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.env = env
        self.gamma = gamma
        self.max_episode_steps = max_episode_steps
        self.device = device
        
        # Prompt template for generating Python programs
        self.prompt_template = """Write a concise Python program that generates the following sequence: {sequence}

The program should be as short as possible while still being correct.
Use functions like range_func_up, set_list, repeat_num, reverse_list, concatenate, etc.
The final result should be assigned to a variable named 'output'.

OUTPUT FORMAT RULES:
1. Generate ONLY ONE LINE of Python code at a time, without explanations
2. Each line must be a valid Python statement (assignment, function call, etc.)
3. Do not include comments, docstrings, or explanations
4. Do not include line numbers, bullets, or other formatting
5. Format should be "variable_name = expression" or similar valid Python syntax
6. Continue generating from where the current program ends

Current program:
{current_program}

Next line of code:"""
        
    def select_action(self, state: str) -> tuple[str, torch.Tensor]:
        """Select next token/line based on current policy.
        
        Args:
            state: Current program state
            
        Returns:
            Selected action and log probability
        """
        # Get sequence from environment
        sequence = self.env.target_sequence
        
        # Create a proper prompt with the target sequence and current program
        prompt = self.prompt_template.format(
            sequence=sequence,
            current_program=state
        )
        
        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Get model predictions
        outputs = self.model(**inputs)
        logits = outputs.logits[0, -1, :]
            
        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        m = Categorical(probs)
        token = m.sample()
        
        # Convert token to code string
        action = self.tokenizer.decode(token)
        
        return action, m.log_prob(token)
        
    def train_step(self, sequence: List[Any]) -> Dict[str, float]:
        """Run one training episode.
        
        Args:
            sequence: Target sequence for the program to output
            
        Returns:
            Training metrics
        """
        # Reset environment
        state = self.env.reset(target_sequence=sequence)
        
        log_probs = []
        rewards = []
        
        # Run episode
        for _ in range(self.max_episode_steps):
            # Select action
            action, log_prob = self.select_action(state)
            log_probs.append(log_prob)
            
            # Take step in environment
            next_state, reward, done, _ = self.env.step(action)
            rewards.append(reward)
            
            if done:
                break
                
            state = next_state
            
        # Calculate returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        policy_loss = torch.stack(policy_loss).sum()
        
        # Optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return {
            "episode_length": len(rewards),
            "total_reward": sum(rewards),
            "policy_loss": policy_loss.item()
        }
    
    def _clean_code_output(self, text: str) -> str:
        """Clean model output to extract only valid Python code.
        
        Args:
            text: Raw model output that may contain explanations or other text
            
        Returns:
            Cleaned code line
        """
        # Remove code markers
        text = text.replace("```python", "").replace("```", "")
        
        # Extract code if there's a specific pattern
        if "=" in text:
            # For assignment statements, try to extract the full line
            parts = text.split("=", 1)
            var_name = parts[0].strip()
            if var_name and not any(x in var_name for x in ["(", ")", "{", "}", "#"]):
                # Looks like a valid variable assignment
                return f"{var_name} = {parts[1].strip()}"
        
        # For function calls or other statements
        # Remove any non-code explanations that might appear before or after the code
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("//") and not line.startswith("/*"):
                # Basic check if it looks like valid Python code
                if any(keyword in line for keyword in ["=", "def ", "if ", "for ", "while ", "import ", "from ", "output "]):
                    return line
        
        # If all else fails, return the original text with minimal cleaning
        return text.strip()
    
    def generate_program(self, sequence: List[int], max_steps: int = 100) -> str:
        """Generate a minimal program that reproduces the given sequence.
        
        Args:
            sequence: Target sequence to reproduce
            max_steps: Maximum number of program generation steps
            
        Returns:
            Generated program as a string
        """
        # Reset environment with target sequence
        state = self.env.reset(target_sequence=sequence)
        done = False
        
        # Track program lines
        program_lines = []
        
        # Generate program step by step
        while not done and len(program_lines) < max_steps:
            # Get model's action (next line of code)
            action, _ = self.select_action(state)
            
            # Clean up the output to extract valid code
            cleaned_action = self._clean_code_output(action)
                
            program_lines.append(cleaned_action)
            
            # Update state and check if done
            state, _, done, info = self.env.step(cleaned_action)
            
            # If done and succeeded, return the program
            if done and info.get("reason") == "success":
                return "\n".join(program_lines)
                
        # If we get here, generation failed
        # Return a simple list-based program as fallback
        return f"output = {sequence}"
