import tiktoken
import requests
from Tok_Reg import RegexTokenizer

# Shared setup
corpus = requests.get("https://tinyurl.com/shakespeare-txt").text[:10000]
test_cases = [
    ("Text with special<|SPECIAL|>token", {"<|SPECIAL|>": 500}, {"<|SPECIAL|>"}),
    ("Numbers: 123,456.78", {}, set()),
    ("Emoji: 😀+🤣", {}, set()),
    (" contractions don't won't", {}, set())
]

# Initialize tokenizers
openai_tokenizer = tiktoken.get_encoding("cl100k_base")
custom_tokenizer = RegexTokenizer()
custom_tokenizer.train(corpus, vocab_size=500)
custom_tokenizer.register_special_tokens({"<|SPECIAL|>": 500})

# Comparison function
def compare_tokenizers(text, specials, allowed):
    print(f"\n{' Test Case: ':-^60}")
    print(f"Text: {text[:50]}...")

    # Custom tokenizer
    custom_tokenizer.register_special_tokens(specials)
    try:
        # Ensure that allowed_special is correctly passed
        custom_encoded = custom_tokenizer.encode(text, allowed_special=allowed)
        custom_decoded = custom_tokenizer.decode(custom_encoded)
        print("\n[Custom Tokenizer]")
        print(f"Tokens: {len(custom_encoded)} | Roundtrip: {text == custom_decoded}")
        print(f"Special tokens: {[t for t in custom_encoded if t in specials.values()]}")
        print(f"Sample tokens: {custom_encoded[:5]}...")
    except Exception as e:
        print(f"Custom Error: {str(e)}")

    # OpenAI tokenizer
    try:
        openai_encoded = openai_tokenizer.encode(text)
        openai_decoded = openai_tokenizer.decode(openai_encoded)
        print("\n[OpenAI Tokenizer]")
        print(f"Tokens: {len(openai_encoded)} | Roundtrip: {text == openai_decoded}")
        print(f"Special preserved: {'<|SPECIAL|>' in openai_decoded if specials else 'N/A'}")
        print(f"Sample tokens: {openai_encoded[:5]}...")
    except Exception as e:
        print(f"OpenAI Error: {str(e)}")

# Run comparisons
for case in test_cases:
    compare_tokenizers(*case)
