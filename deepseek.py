import psutil
from llama_cpp import Llama


class DeepSeek:
    def __init__(self, model_path):

        self.model = Llama(model_path=model_path, n_ctx=2048, n_batch=64, verbose=False)

    def generate_response(self, prompt, temperature=0.7, max_tokens=250, top_p=0.9):
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