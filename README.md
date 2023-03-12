# text-guided-diffusion-style-transfer

This is my attempt at implementing [Text-Guided Diffusion Image Style Transfer with Contrastive Loss Fine-tuning](https://openreview.net/forum?id=iJ_E0ZCy8fi).

Here is some stuff I need to take into account:
  - They use DDIM as forward and DDPM as reverse process, in particular, they utilize the [unconditional diffusion model trained on ImageNET](https://github.com/openai/guided-diffusion) dataset with 256 × 256 image size (Dhariwal & Nichol, 2021) and the model trained on FFHQ dataset with 256 × 256 image size (Choi et al., 2021).
  - They use Wikiart dataset to evaluate the model.
  - All the images are resized to 256 × 256 for the diffusion models. For patch-based guidance, they randomly crop 96 patches from a source image and then apply perspective augmentation and affine transformation.
  - Reference of a [DDPM + CLIP notebook](https://colab.research.google.com/drive/1V66mUeJbXrTuQITvJunvnWVn96FEbSI3#scrollTo=X5gODNAMEUCR).
  - Code [reference of directional loss](https://github.com/rinongal/StyleGAN-nada/blob/main/ZSSGAN/criteria/clip_loss.py) (eq 8).
  - [Paper reference for CUT loss](https://arxiv.org/abs/2007.15651).
  - They mention a noise estimator to exctract features from, it's basically the unet.

---

## Setup project

Install dependencies:

```console
pip install -e ./CLIP & pip install -e ./guided-diffusion
```

To download the unconditional diffusion model, run (it weights 2.06GB):

```console
wget -O unconditional_diffusion.pt https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
```

If all went smooth, then you should be able to run `main.py` without any troubles.
