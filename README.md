# Cria

Cria is a small-scale, LLaMA-style transformer being built to understand how modern transformer-based language models work.

---

## Planned Architecture Components

The model will include the following components:

- Token embedding layer
- Rotary positional embeddings (RoPE)
- Transformer blocks:
  - RMS-based normalization
  - Multi-head self-attention
  - Residual connection
  - Feed-forward network with SwiGLU activation
  - Residual connection
- Final projection to vocabulary logits

---

## Current Status

- Architecture planning and reference reading in progress
- Modules under active development

The repository will be updated incrementally as components are implemented and tested.

---

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I.  
  *Attention Is All You Need* (2017)

- Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Joulin, A., Grave, E., and Lample, G.  
  *LLaMA: Open and Efficient Foundation Language Models* (2023)

---

## Disclaimer

Cria is a learning project developed as part of undergraduate self-study.  
It is not intended to reproduce, replace, or compete with any existing large-scale language model.
