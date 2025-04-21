import psutil
from llama_cpp import Llama
def get_optimal_llm_params():
    """
    Dynamically sets LLM parameters based on system resources.
    """
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    cpu_cores = psutil.cpu_count(logical=False)

    # Set n_ctx dynamically
    if total_ram_gb <= 8:
        _n_ctx = 1024
    elif total_ram_gb <= 16:
        _n_ctx = 2048
    else:
        _n_ctx = 4096  # High RAM systems

    # Set n_batch dynamically (MINIMUM 64)
    if cpu_cores <= 4:
        _n_batch = 64  # Increased from 8
    elif cpu_cores <= 8:
        _n_batch = 128  # Increased from 16
    else:
        _n_batch = 256  # Increased from 32 for high-performance CPUs

    return _n_ctx, _n_batch


n_ctx, n_batch = get_optimal_llm_params()

class Phi2LLM:
    def __init__(self, model_path="models/phi-2-q4_k_m.gguf"):

        self.model = Llama(model_path=model_path, n_ctx=n_ctx, n_batch=n_batch, verbose=False)

    def generate_response(self, prompt, temperature=0.7, max_tokens=100, top_p=0.9):
        """
        Generate a response from Phi-2 with tunable hyperparameters.
        :param prompt: The input prompt.
        :param temperature: Controls randomness (0.0 = deterministic, 1.0 = very random).
        :param max_tokens: Maximum number of tokens in output.
        :param top_p: Controls nucleus sampling.
        """
        response = self.model(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            repeat_penalty=1.2  # Reduces repetitive answers

        )
        return response["choices"][0]["text"]