JAX implementation of PARK block retrieval, arXiv:2609.38978, PAMA equations 8-9, QRKC Appendix F Algorithm 3.

```bash
pip install -e .
pytest
python train.py
```

No HunyuanVideo or Wan models, no fused CUDA or Triton kernel, and no claimed speedups or quality numbers. Shrinkage alpha defaults to 0.05 as stated in the paper. The stability constant is not given a number there, so eps defaults to 1e-6.
