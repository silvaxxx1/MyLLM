"""Entry point for `python -m myllm` and the `myllm` CLI command."""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="myllm",
        description="myllm — from-scratch LLM framework",
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("version", help="Print version and exit")
    sub.add_parser("models", help="List available model configs")

    info_p = sub.add_parser("info", help="Show memory estimate for a model")
    info_p.add_argument("model", help="e.g. gpt2-small, gpt2-medium, llama3-8b")

    args = parser.parse_args()

    if args.cmd == "version":
        from myllm import __version__
        print(f"myllm {__version__}")

    elif args.cmd == "models":
        from myllm import ModelConfig
        print("Available model configs:")
        for name in ModelConfig.available_configs():
            print(f"  {name}")

    elif args.cmd == "info":
        import torch
        from myllm import ModelConfig
        cfg = ModelConfig.from_name(args.model)
        mem = cfg.estimate_memory(batch_size=1, dtype=torch.float32)
        print(f"\nModel : {args.model}")
        print(f"Layers: {cfg.n_layer}  Heads: {cfg.n_head}  Embd: {cfg.n_embd}")
        print(f"Params: {mem['n_parameters'] / 1e6:.1f}M")
        print(f"Memory (fp32): {mem['parameters_gb']:.2f} GB params  "
              f"+ {mem['activations_gb']:.2f} GB activations")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
