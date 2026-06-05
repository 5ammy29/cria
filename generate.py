import torch
import sentencepiece as spm

from model import Config, Transformer


def main():

    sp = spm.SentencePieceProcessor()
    sp.load("tokenizer.model")

    config = Config(
        vocab_size=sp.vocab_size()
    )

    model = Transformer(config)

    checkpoint = torch.load(
        "best.pt",
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    prompt = "MARCIUS:"

    tokens = sp.encode(prompt)

    tokens = torch.tensor(
        [tokens],
        dtype=torch.long
    )

    with torch.no_grad():

        for _ in range(100):

            x = tokens[:, -config.seq_len:]

            logits = model(x)

            next_logits = logits[:, -1, :]

            probs = torch.softmax(
                next_logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            tokens = torch.cat(
                [tokens, next_token],
                dim=1
            )

    text = sp.decode(
        tokens[0].tolist()
    )

    print("\n" + "=" * 50)
    print(text)
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()