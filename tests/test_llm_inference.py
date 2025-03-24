import pytest
import subprocess
from src.llm_inference import call_ollama

class FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: str, stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

def fake_run_success(command, capture_output, text):
    return FakeCompletedProcess(returncode=0, stdout="Fake LLM Answer")

def fake_run_failure(command, capture_output, text):
    return FakeCompletedProcess(returncode=1, stdout="", stderr="Error occurred")

def test_call_ollama_success(monkeypatch):
    monkeypatch.setattr(subprocess, "run", fake_run_success)
    prompt = "Test prompt"
    answer = call_ollama(prompt, model="Mistral")
    assert answer == "Fake LLM Answer"

def test_call_ollama_failure(monkeypatch):
    monkeypatch.setattr(subprocess, "run", fake_run_failure)
    prompt = "Test prompt"
    with pytest.raises(RuntimeError) as excinfo:
        call_ollama(prompt, model="Mistral")
    assert "Ollama call failed" in str(excinfo.value)
