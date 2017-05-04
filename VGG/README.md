Data preprocessing:

```bash
OMP_NUM_THREADS=2 th -i provider.lua
```

```lua
provider = Provider()
provider:normalize()
torch.save('provider.dat',provider)
```

Training:

```bash
CUDA_VISIBLE_DEVICES=0 th train.lua --model vgg_bn_drop -s logs/vgg --backend cudnn
```