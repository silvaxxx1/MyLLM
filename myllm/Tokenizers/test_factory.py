# test_factory.py
from myllm.Tokenizers.factory import get_tokenizer, list_available_models

def test_tokenizer():
    print("=== Single Tokenizer Test ===")
    tok = get_tokenizer("gpt2")
    print(tok)  # TokenizerWrapper representation

    text = "Hello world!"
    ids = tok.encode(text)
    print("Single encode:", ids)
    print("Single decode:", tok.decode(ids))

    print("\n=== Batch Encode Test ===")
    batch_sentences = ["Hello world!", "This is a test sentence.", "Another one."]
    try:
        batch_encoded = tok.batch_encode(batch_sentences, padding=True, return_tensors="pt")
        print("Batch input_ids:\n", batch_encoded["input_ids"])
        print("Batch attention_mask:\n", batch_encoded["attention_mask"])
    except AttributeError:
        # Fallback: encode individually
        batch_encoded = [tok.encode(s) for s in batch_sentences]
        print("Batch encoded (list):", batch_encoded)

    print("\n=== Caching Test ===")
    tok2 = get_tokenizer("gpt2")
    print("Same instance:", tok is tok2)

    print("\n=== Available Models ===")
    print(list_available_models())

if __name__ == "__main__":
    test_tokenizer()
