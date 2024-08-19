

---

# Deep Disentangled Metric Learning

This is the official PyTorch implementation of [**Deep Disentangled Metric Learning**]. This repository provides the source code for experiments conducted on three datasets (CUB-200-2011, Cars-196, Stanford Online Products) and pretrained models with BN-Inception as the backbone.

---

## Requirements

Ensure the following dependencies are installed:

- Python 3
- PyTorch (> 1.0)
- NumPy
- tqdm
- wandb
- [Pytorch-Metric-Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)

---

## Datasets

1. Download the following public benchmarks for deep metric learning:
   - [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
   - Cars-196 ([Images](http://imagenet.stanford.edu/internal/car196/car_ims.tgz), [Annotations](http://imagenet.stanford.edu/internal/car196/cars_annos.mat))
   - [Stanford Online Products](https://cvgl.stanford.edu/projects/lifted_struct/)

2. Extract the downloaded `.tgz` or `.zip` files into the `./data/` directory.

---

## Training Embedding Network

### CUB-200-2011

Train an embedding network of Inception-BN (d=512) using **DDML(+PA)**:

```bash
python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cub \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10 \
                --disent True \
                --al 1.0 \
                --be 0.1 \
                --gam 0.00001
```

| Method | Backbone | R@1 | R@2 | R@4 | R@8 |
|:------:|:--------:|:---:|:---:|:---:|:---:|
| [DDML(+PA)<sup>512</sup>] | Inception-BN | 70.0 | 79.6 | 87.1 | 92.0 |

---

### Cars-196

Train an embedding network of Inception-BN (d=512) using **DDML(+PA)**:

```bash
python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset car \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 20 \
                --disent True \
                --al 1.0 \
                --be 0.1 \
                --gam 0.001
```

| Method | Backbone | R@1 | R@2 | R@4 | R@8 |
|:------:|:--------:|:---:|:---:|:---:|:---:|
| [DDML(+PA)<sup>512</sup>] | Inception-BN | 87.9 | 92.6 | 95.7 | 97.4 |

---

### Stanford Online Products

Train an embedding network of Inception-BN (d=512) using **Proxy-Anchor loss**:

```bash
python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 6e-4 \
                --dataset SOP \
                --warm 1 \
                --bn-freeze 0 \
                --lr-decay-step 20 \
                --lr-decay-gamma 0.25 \
                --disent True \
                --al 1.0 \
                --be 1.0 \
                --gam 0.0001
```

| Method | Backbone | R@1 | R@10 | R@100 | R@1000 |
|:------:|:--------:|:---:|:----:|:-----:|:------:|
| [DDML(+PA)<sup>512</sup>] | Inception-BN | 79.9 | 91.1 | 96.2 | 98.6 |

---

## Acknowledgements

This code is modified and adapted from the following great repositories:

- [Proxy Anchor Loss for Deep Metric Learning](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020)
- [No Fuss Distance Metric Learning using Proxies](https://github.com/dichotomies/proxy-nca)
- [PyTorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)

---

