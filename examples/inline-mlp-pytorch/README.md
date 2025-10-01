# Inline MLP (SlangPy + PyTorch)

SlangPy port of the inline MLP image fitting idea (no slang-torch bindings).

## Contents
- `inline_mlp_itensor.slang`: per-pixel render entry (`render_image`) using an ITensor feature grid and a tiny MLP.
- `main.py`: PyTorch loop optimizing weights/biases/feature grid to match a simple target.

## Run
```bash
python main.py
```
Generates `inline_mlp_pytorch.mp4` showing prediction, target, and error.
