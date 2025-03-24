import subprocess
import os

def call_ollama(prompt: str, model: str = "Mistral") -> str:
    """
    Call Ollama CLI to perform LLM inference with a given prompt.
    
    Args:
        prompt (str): The prompt text.
        model (str): The LLM model name (e.g., "Mistral").
        
    Returns:
        str: The LLM-generated answer.
        
    Raises:
        RuntimeError: If the Ollama call fails.
    """
    command = ["ollama", "run", model, prompt]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Ollama call failed: {result.stderr}")
    return result.stdout.strip()