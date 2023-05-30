# text-guided-diffusion-style-transfer

This is my attempt at implementing [Zero-Shot Contrastive Loss for Text-Guided Diffusion Image Style Transfer ](https://arxiv.org/abs/2303.08622).

## Roadmap

- [x] Setup repo with initial dependencies (CLIP + guided diffusion).
- [x] Add CLIP guidance to unconditional 256 model.
- [x] Add CLIP guidance over inverted image.
- [x] Add global loss.
- [x] Add directional loss.
- [x] Add patch-based loss
- [x] Add feature loss.
- [x] Add pixel loss.
- [x] Add ZeCon loss.
- [ ] Test with Wikiart dataset.

## Setup project

Clone submodules:

```
git clone https://github.com/openai/CLIP
git clone https://github.com/ouhenio/guided-diffusion.git
```

Install submodules dependencies:

```console
pip install -e ./CLIP & pip install -e ./guided-diffusion
```

Download the unconditional diffusion model (weights 2.06GB):

```console
wget -O unconditional_diffusion.pt https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
```

## Sampling

Run

```console
python main.py
```

---

Here is some stuff I need to take into account:
  - They use DDIM as forward and DDPM as reverse process, in particular, they utilize the [unconditional diffusion model trained on ImageNET](https://github.com/openai/guided-diffusion) dataset with 256 × 256 image size (Dhariwal & Nichol, 2021) and the model trained on FFHQ dataset with 256 × 256 image size (Choi et al., 2021).
  - They use Wikiart dataset to evaluate the model.
  - All the images are resized to 256 × 256 for the diffusion models. For patch-based guidance, they randomly crop 96 patches from a source image and then apply perspective augmentation and affine transformation.
  - Reference of a [DDPM + CLIP notebook](https://colab.research.google.com/drive/1V66mUeJbXrTuQITvJunvnWVn96FEbSI3#scrollTo=X5gODNAMEUCR).
  - Code [reference of directional loss](https://github.com/rinongal/StyleGAN-nada/blob/main/ZSSGAN/criteria/clip_loss.py) (eq 8).
  - Rreference of the [patch-based CLIP loss ](https://github.com/cyclomon/CLIPstyler) they applied to global an directional loss.
  - [Paper reference for CUT loss](https://arxiv.org/abs/2007.15651).
  - They mention a noise estimator to exctract features from, it's basically the unet.
  - [Reference to global clip loss mentioned in the paper](https://github.com/orpatashnik/StyleCLIP/blob/main/criteria/clip_loss.py)
  - [Reference of unet feature exctractor](https://github.com/yandex-research/ddpm-segmentation/blob/master/src/feature_extractors.py)
