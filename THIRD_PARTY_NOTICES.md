# Third-party components

- **VadCLIP**: https://github.com/nwpu-zxr/VadCLIP, Apache License 2.0.
  The two-branch backbone, graph layers, prompt encoding, MIL objectives,
  temporal sampling, and temporal detection metrics derive from this project.
  Changes in this repository add AGRC and VLSR, portable configuration and
  two-stage entry points, device-aware graph operations, and category readout.
  The Apache license is included in `LICENSE`.
- **OpenAI CLIP**: https://github.com/openai/CLIP, MIT License.
  The bundled `src/clip` implementation and tokenizer vocabulary come through
  VadCLIP and support its learnable prompt-embedding interface. See
  `licenses/CLIP.txt` for the original copyright and license.
- **DSANet**: https://github.com/lessiYin/DSANet, MIT License.
  The hierarchical probability factorization in `src/evaluate.py` is adapted
  from DSANet, with an interpolation with native category probabilities.
  See `licenses/DSANet.txt` for the original copyright and license.

Dataset annotations and pre-extracted features are obtained separately from
their original providers. Their respective terms continue to apply.
