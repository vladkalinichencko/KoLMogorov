"""
Environment for training language models to minimize program length using REINFORCE.
"""
import ast
import sys
from typing import Dict, List, Tuple, Optional
import numpy as np

class KolmogorovEnv:
    def __init__(self, reward_constant: float = 100.0, length_penalty: float = 0.1, max_steps: int = 100):
        """Initialize the environment.
        
        Args:
            reward_constant: Base reward for correct programs (C in the paper)
            length_penalty: Penalty coefficient for program length (λ in the paper)
            max_steps: Maximum number of steps before termination
        """
        self.reward_constant = reward_constant
        self.length_penalty = length_penalty
        self.max_steps = max_steps
        self.reset()
    
    def reset(self, target_sequence: Optional[List] = None) -> str:
        """Reset the environment with a new target sequence.
        
        Args:
            target_sequence: The sequence that the program should output
            
        Returns:
            Initial state (empty program)
        """
        self.target_sequence = target_sequence
        self.current_program = ""
        self.steps = 0
        return self.current_program
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Take a step in the environment by adding a token/line.
        
        Args:
            action: Token/line to add to the program
            
        Returns:
            (new_state, reward, done, info)
        """
        self.steps += 1
        done = False
        
        # Add the new token/line
        if not self.current_program.strip():
            self.current_program = action
        else:
            self.current_program = self.current_program + "\n" + action
            
        # Check if max steps reached
        if self.steps >= self.max_steps:
            return self.current_program, 0.0, True, {"reason": "max_steps"}
            
        # Try to run the program and check output
        try:
            # Add output variable if not present
            if "output = " not in self.current_program:
                prog = self.current_program + "\noutput = None"
            else:
                prog = self.current_program
                
            # Create safe locals dict and execute
            local_vars = {}
            exec(prog, {"__builtins__": {}}, local_vars)
            
            # Check if output matches target
            if self.target_sequence is not None and "output" in local_vars:
                if local_vars["output"] == self.target_sequence:
                    # Calculate encoded length (simple for now)
                    encoded_length = len(self.current_program)
                    reward = self.reward_constant - self.length_penalty * encoded_length
                    done = True
                    return self.current_program, reward, done, {"reason": "success"}
                    
        except Exception:
            # Program error
            pass
            
        # If we get here, program is not complete/correct
        return self.current_program, 0.0, done, {"reason": "incomplete"}
