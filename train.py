import modal
import torch
import torch.nn.functional as F
import sentencepiece as spm

app = modal.App("tiny-transformer")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "sentencepiece",
        "numpy"
    )
    .add_local_dir(".", "/root/cria")
)

volume = modal.Volume.from_name(
    "transformer-checkpoints",
    create_if_missing=True,
)

ROOT = "/data"

def get_batch(data, batch_size, seq_len):

    starts = torch.randint(
        0,
        len(data) - seq_len - 1,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + seq_len]
        for i in starts
    ])

    y = torch.stack([
        data[i + 1:i + seq_len + 1]
        for i in starts
    ])

    return x, y

@torch.no_grad()
def estimate_loss(
    model,
    data,
    config,
    batch_size,
    device,
    eval_iters=20,
):

    model.eval()

    losses = []

    for i in range(eval_iters):

        x, y = get_batch(
            data,
            batch_size,
            config.seq_len
        )

        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                config.vocab_size
            ),
            y.reshape(-1)
        )

        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)

@app.function(
    image=image,
    gpu="A10G",
    cpu=4,
    memory=16000,
    timeout=60 * 60 * 10,
    volumes={ROOT: volume},
)
def train():

    import sys

    sys.path.append("/root/cria")

    from model import Config, Transformer

    device = torch.device("cuda")

    sp = spm.SentencePieceProcessor()
    sp.load("/root/cria/tokenizer.model")

    with open(
        "/root/cria/tinyshakespeare.txt",
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    tokens = sp.encode(text)

    tokens = torch.tensor(
        tokens,
        dtype=torch.long
    )

    n = int(0.9 * len(tokens))

    train_data = tokens[:n]
    val_data = tokens[n:]

    config = Config(
        vocab_size=sp.vocab_size()
    )

    model = Transformer(config).to(device)

    print(
        "Parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01
    )

    batch_size = 32
    max_steps = 10000

    best_val_loss = float("inf")

    model.train()

    for step in range(max_steps):

        x, y = get_batch(
            train_data,
            batch_size,
            config.seq_len
        )

        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                config.vocab_size
            ),
            y.reshape(-1)
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        if step % 100 == 0:

            val_loss = estimate_loss(
                model,
                val_data,
                config,
                batch_size,
                device
            )

            print(
                f"step={step} "
                f"train_loss={loss.item():.4f} "
                f"val_loss={val_loss:.4f}"
            )

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "val_loss": val_loss,
                    },
                    f"{ROOT}/best.pt"
                )

                volume.commit()

        if step % 1000 == 0 and step > 0:

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                },
                f"{ROOT}/checkpoint_{step}.pt"
            )

            volume.commit()

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": max_steps,
            "best_val_loss": best_val_loss,
        },
        f"{ROOT}/shakespeare.pt"
    )

    volume.commit()

@app.local_entrypoint()
def main():
    train.remote()