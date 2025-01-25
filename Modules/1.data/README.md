# MyLLM Data Directory

This directory contains scripts and utilities for handling and processing data used in training a GPT-like language model. It includes functionalities for data loading, preprocessing, and custom tokenization.

## Directory Structure

```
MyLLM/
└── data/
    ├── dataloader.py           # DataLoader for the GPTDataset
    ├── preprocess.py            # Functions for preprocessing raw data
    ├── tokenizer/               # Tokenization scripts and utilities
    ├── data_test.py             # Tests for data processing scripts
    └── tests/                   # Unit tests for dataset and preprocessing functions
        ├── test_gpt_dataset.py  # Tests for GPTDataset class
        └── test_preprocess.py    # Tests for preprocessing functions
```

## Usage

- **Dataset and DataLoader**: Use the `GPTDataset` and `GPTDataLoader` classes to prepare data for model training.
- **Preprocessing**: The `preprocess.py` script contains functions for cleaning and preparing raw data.
- **Tokenization**: The `tokenizer/` directory includes a custom tokenizer and functionality for handling special tokens, along with comparisons to OpenAI's `tiktoken`.

## Tokenizer

The `tokenizer/` module provides custom text tokenization functionality essential for preparing text data for NLP tasks. It includes a custom tokenizer based on regular expressions, special token handling, and comparisons with OpenAI's `tiktoken`.

### Key Features:
1. **Custom Tokenizer**: The `RegexTokenizer` class, implemented in `Tok_Reg.py`, breaks text into tokens using regular expressions and can be trained on a given corpus.
2. **Special Token Handling**: Allows registering special tokens (e.g., `<|SPECIAL|>`).
3. **Comparison with OpenAI Tokenizer**: A function to compare the performance of the custom tokenizer with OpenAI’s tokenizer.

### How It Works:
1. **Fetch a Corpus**: The script fetches a text corpus (e.g., Shakespeare’s works) and truncates it to the first 10,000 characters.
   ```python
   corpus = requests.get("https://tinyurl.com/shakespeare-txt").text[:10000]
   ```
2. **Train the Tokenizer**: The custom tokenizer is trained on the fetched corpus, and special tokens can be registered.
   ```python
   custom_tokenizer = RegexTokenizer()
   custom_tokenizer.train(corpus, vocab_size=500)
   custom_tokenizer.register_special_tokens({"<|SPECIAL|>": 500})
   ```
3. **Run Test Cases**: Test cases evaluate tokenization across different types of input, including special tokens, numbers, emojis, and contractions.
   ```python
   test_cases = [
       ("Text with special<|SPECIAL|>token", {"<|SPECIAL|>": 500}, {"<|SPECIAL|>"}),
       ("Numbers: 123,456.78", {}, set()),
       ("Emoji: 😀+🤣", {}, set()),
       (" contractions don't won't", {}, set())
   ]
   for case in test_cases:
       compare_tokenizers(*case)
   ```
4. **Comparison with OpenAI Tokenizer**: The behavior of the custom tokenizer is compared with OpenAI's `tiktoken` tokenizer.
   ```python
   openai_tokenizer = tiktoken.get_encoding("cl100k_base")
   ```

### Running the Script
To run the script that tests and compares the tokenizers, simply execute the `test_script.py`:

```bash
python tokenizer/test_script.py
```

This will run the test cases and print the results comparing both tokenizers.

## Testing

To run tests for data processing and tokenization, use:

```bash
python -m unittest discover -s tests
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```