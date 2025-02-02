import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import tiktoken


class TextGenerator:
    def __init__(self, model_name="gpt2", device=None):
        """
        Initializes the text generator.

        Args:
            model_name (str): Pretrained model name (default: "gpt2").
            device (str, optional): "cuda" or "cpu". Auto-detects if not provided.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoding = tiktoken.get_encoding("gpt2")  # Using tiktoken for fast encoding

    def get_log_prob(self, logits, token_id):
        """Compute log probability of a given token in logits."""
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        log_prob = torch.log(probabilities[token_id])
        return log_prob.item()

    def greedy_sampling(self, logits, beams):
        """Greedy sampling: Selects top-k highest probability tokens."""
        return torch.topk(logits, beams).indices

    def top_k_sampling(self, logits, temperature, top_k, beams):
        """Top-k sampling: Selects from the top-k highest probability tokens."""
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        probabilities = torch.nn.functional.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probabilities, beams)

    def nucleus_sampling(self, logits, temperature, p, beams):
        """Nucleus (top-p) sampling: Selects from a minimal set of tokens that sum to probability >= p."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probabilities = torch.nn.functional.softmax(sorted_logits / temperature, dim=-1)
        cumulative_probabilities = torch.cumsum(probabilities, dim=-1)

        # Masking tokens whose cumulative probability is below p
        sorted_indices_to_remove = cumulative_probabilities > p
        sorted_logits[sorted_indices_to_remove] = float('-inf')

        probabilities = torch.nn.functional.softmax(sorted_logits / temperature, dim=-1)
        return torch.multinomial(probabilities, beams)

    def generate(self, prompt, length=50, beams=3, sampling="greedy", temperature=0.7):
        """
        Generates text based on a given prompt.

        Args:
            prompt (str): Input text prompt.
            length (int): Number of tokens to generate.
            beams (int): Number of beams for beam search.
            sampling (str): Sampling method ('greedy', 'top_k', 'nucleus').
            temperature (float): Softmax temperature for sampling.

        Returns:
            str: Generated text.
        """
        # Tokenize input prompt
        input_ids = torch.tensor(self.tokenizer.encode(prompt, return_tensors="pt")).to(self.device)

        # Initialize sequence list with the input tokens and score (log probability)
        sequences = [(input_ids, 0)]  # (sequence, score)

        for _ in range(length):
            new_sequences = []

            # Iterate through current sequences
            for seq, score in sequences:
                # Get model outputs and extract logits for the last token
                outputs = self.model(seq)
                logits = outputs.logits[0, -1, :]

                # Select the next token(s) based on the chosen sampling method
                if sampling == 'greedy':
                    top_token_ids = self.greedy_sampling(logits, beams)
                elif sampling == 'top_k':
                    top_token_ids = self.top_k_sampling(logits, temperature, 20, beams)
                elif sampling == 'nucleus':
                    top_token_ids = self.nucleus_sampling(logits, temperature, 0.5, beams)
                else:
                    raise ValueError(f"Unsupported sampling method: {sampling}")

                # Iterate through the selected token candidates
                for token_id in top_token_ids:
                    # Compute log probability of the selected token
                    token_score = self.get_log_prob(logits, token_id)
                    cumulative_score = score + token_score  # Update sequence score

                    # Append the new token to the sequence
                    new_input_ids = torch.cat([seq, token_id.unsqueeze(0).unsqueeze(0)], dim=-1)
                    new_sequences.append((new_input_ids, cumulative_score))

            # Keep only the top `beams` sequences based on score
            sequences = sorted(new_sequences, key=lambda x: x[1], reverse=True)[:beams]

        # Decode the best sequence into text
        best_sequence = sequences[0][0].squeeze().tolist()
        return self.tokenizer.decode(best_sequence, skip_special_tokens=True)
