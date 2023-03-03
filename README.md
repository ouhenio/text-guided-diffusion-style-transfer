# text-guided-diffusion-style-transfer

My attempt at implementing [Text-Guided Diffusion Image Style Transfer with Contrastive Loss Fine-tuning](https://openreview.net/forum?id=iJ_E0ZCy8fi).

Some stuff to take into account:
  - They use DDIM as forward and DDPM as reverse process, in particular, they utilize the unconditional diffusion model trained on ImageNET dataset with 256 × 256 image size (Dhariwal & Nichol, 2021) and the model trained on FFHQ dataset with 256 × 256 image size (Choi et al., 2021).
  - Furthermore, in order to evaluate the performance of our proposed model on the images from unseen domains, we utilize Wikiart dataset.
  - All the images are resized to 256 × 256 for the diffusion models. For patch-based guidance, we randomly crop 96 patches from a source image and then apply perspective augmentation and affine transformation.
