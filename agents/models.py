from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


class HFModel(ABC):
    """
    Abstract base class for Hugging Face language models.
    Subclasses must implement 'generate_response'.
    """

    def __init__(self, checkpoint: str, device: str = "cpu") -> None:
        """
        Initialize a Hugging Face model and tokenizer.

        :param checkpoint: The model checkpoint name or path (from Hugging Face Hub).
        :param device: The device on which to load the model ('cpu' or 'cuda').
        """
        self.checkpoint = checkpoint
        self.device = device
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    @abstractmethod
    def generate_response(
            self,
            messages: List[Dict[str, str]],
            max_new_tokens: int = 100,
            temperature: float = 0.2,
            top_p: float = 0.9,
            top_k: int = 50,
            do_sample: bool = True,
            **kwargs: Any
    ) -> str:
        """
        Subclasses must define how to build input data from 'messages'
        and produce a response string.

        :param messages: A list of dicts representing a chat or conversation context.
               Each dict has "role" and "content" keys, for example:
               [{"role": "user", "content": "Hello!"}, ...]
        :param max_new_tokens: Maximum number of new tokens to generate in the response.
        :param temperature: The temperature of sampling. Higher values = more random.
        :param top_p: The cumulative probability for nucleus sampling.
        :param top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
        :param do_sample: Whether or not to use sampling; use greedy decoding otherwise.
        :param kwargs: Additional model.generate() parameters as needed.
        :return: The generated text response as a string.
        """
        pass


class SmollLLM(HFModel):

    def generate_response(
            self,
            messages: List[Dict[str, str]],
            max_new_tokens: int = 50,
            temperature: float = 0.7,
            top_p: float = 0.9,
            top_k: int = 50,
            do_sample: bool = True,
            **kwargs: Any
    ) -> str:
        """
        Generates a text response given a list of chat-like messages.

        :param messages: A list of { "role": "system"/"user"/"assistant", "content": str }.
        :param max_new_tokens: Max number of new tokens to generate in the response.
        :param temperature: Sampling temperature, higher = more random.
        :param top_p: Nucleus sampling probability cutoff.
        :param top_k: Top-k filtering cutoff.
        :param do_sample: Whether or not to sample (True) or do greedy decode (False).
        :param kwargs: Additional parameters to pass to model.generate().
        :return: The generated text as a string.
        """
        # 1) Build the chat prompt for SmolLM. The custom method
        #    'apply_chat_template' helps format messages into a single prompt.
        input_text: str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 2) Tokenize the prompt
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

        # 3) Generate output with the provided generation parameters
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                **kwargs
            )

        input_length = inputs.shape[-1]
        generated_tokens = outputs[0][input_length:]

        # 4) Decode the tokens to a string
        generated_text: str = self.tokenizer.decode(generated_tokens)

        return generated_text


if __name__ == "__main__":
    checkpoint: str = "HuggingFaceTB/SmolLM-135M-Instruct"
    device: str = "cpu"  # or "cpu"

    llm: HFModel = SmollLLM(checkpoint, device=device)

    messages: List[Dict[str, str]] = [
        {"role": "user", "content": "What is the capital of France?"}
    ]

    response: str = llm.generate_response(
        messages,
        max_new_tokens=50,
        temperature=0.2,
        top_p=0.9,
        top_k=50,
        do_sample=True
    )

    print("\n===== Model Response =====")
    print(response)
