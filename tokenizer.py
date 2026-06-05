import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="tinyshakespeare.txt",
    model_prefix="tokenizer",
    vocab_size=1024,
    model_type="bpe",
    character_coverage=1.0
)
