# GPT2-ZeRO1

This is a JAX implementation of GPT-2 with ZeRO-1 (see [the paper](https://arxiv.org/pdf/1910.02054) for more details). In ZeRO-1, we shard the optimizer state while leaving parameters, gradients, and activations unsharded.

To run the training script:
```
cd GPT2-DDP/gpt2ddp/gpt2ddp
uv run scripts/train.py
```

To modify the model/training configuration, see [`gpt2ddp/core/config.py`](gpt2ddp/core/config.py).

Here's a memory profile of 16 training steps. Compared to my [experiments with GPT2-DDP](https://github.com/TheBatmanofButler/gpt2-ddp), we see ~800 MB reduction in max memory use:
<img width="1146" height="755" alt="image" src="https://github.com/user-attachments/assets/09afbcd5-79d9-49b3-84d9-fef9cc2f1b2b" />
