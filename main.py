import clip
import torch
import tqdm

from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

model_config = model_and_diffusion_defaults()
model_config.update({
    'attention_resolutions': '32, 16, 8',
    'class_cond': False,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing': '1000',
    'image_size': 256,
    'learn_sigma': True,
    'noise_schedule': 'linear',
    'num_channels': 256,
    'num_head_channels': 64,
    'num_res_blocks': 2,
    'resblock_updown': True,
    'use_fp16': True,
    'use_scale_shift_norm': True,
})

# Load models

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model, diffusion = create_model_and_diffusion(**model_config)
model.load_state_dict(torch.load('unconditional_diffusion.pt', map_location='cpu'))
model.requires_grad_(False).eval().to(device)
for name, param in model.named_parameters():
    if 'qkv' in name or 'norm' in name or 'proj' in name:
        param.requires_grad_()
if model_config['use_fp16']:
    model.convert_to_fp16()

clip_model, clip_preprocess = clip.load('ViT-B/16', jit=False)
clip_model = clip_model.eval().requires_grad_(False).to(device)
clip_size = clip_model.visual.input_resolution
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

# Define clip-related functions

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])

def directional_loss(x, x_t, p_source, p_target):
    # x = clip_preprocess(x).unsqueeze(0).to(device)
    # x = clip_model.encode_image(x).float()
    # x_t = clip_preprocess(x_t).unsqueeze(0).to(device)
    # x_t = clip_model.encode_image(x_t).float()
    # p_source = clip_model.encode_text(clip.tokenize(p_source).to(device)).float()
    # p_target = clip_model.encode_text(clip.tokenize(p_target).to(device)).float()
    img_diff = x - x_t
    text_diff = p_source - p_target
    # todo: check f this is the correct way to compute the value
    norm = torch.matmul(img_diff.view(1,-1), text_diff.view(-1,1)) / (torch.norm(img_diff) * torch.norm(text_diff))
    return 1 - norm

# Run clip-guided diffusion

p_source = "painting"
p_target = "pixar"
batch_size = 1
clip_guidance_scale = 1
tv_scale = 150
skip_timesteps = 500 # this should have a value between 200-500 when using a init img
cutn = 42
cut_pow = 0.5
n_batches = 1
seed = 17

if seed is not None:
    torch.manual_seed(seed)

text_embed_source = clip_model.encode_text(clip.tokenize(p_source).to(device)).float()
text_embed_target = clip_model.encode_text(clip.tokenize(p_target).to(device)).float()
make_cutouts = MakeCutouts(clip_size, cutn, cut_pow)
cur_t = None

init_image_path = "elin.jpg"
init_image = Image.open(init_image_path).convert('RGB')
init_image = init_image.resize((model_config['image_size'], model_config['image_size']), Image.LANCZOS)
init_image_embedding = clip_preprocess(init_image).unsqueeze(0).to(device)
init_image_embedding = clip_model.encode_image(init_image_embedding).float()
init_image_tensor = TF.to_tensor(init_image).to(device).unsqueeze(0).mul(2).sub(1)

if model_config['timestep_respacing'].startswith('ddim'):
    sample_fn = diffusion.ddim_sample_loop_progressive
else:
    sample_fn = diffusion.p_sample_loop_progressive

# Conditionin function

def cond_fn(x, t, y=None):
    with torch.enable_grad():
        x = x.detach().requires_grad_()
        n = x.shape[0]
        my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
        out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
        fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
        x_in = out['pred_xstart'] * fac + x * (1 - fac)
        resized_x_in = F.interpolate(x_in, size=(clip_size, clip_size), mode='bilinear', align_corners=False)
        x_t = clip_model.encode_image(resized_x_in).float()
        clip_in = normalize(make_cutouts(x_in.add(1).div(2)))
        image_embeds = clip_model.encode_image(clip_in).float().view([cutn, n, -1])
        dists = spherical_dist_loss(image_embeds, text_embed_target.unsqueeze(0))
        losses = dists.mean(0)
        dir_loss = directional_loss(init_image_embedding, x_t, text_embed_source, text_embed_target)
        # original loss
        # tv_losses = tv_loss(x_in)
        # loss = losses.sum() * clip_guidance_scale + tv_losses.sum() * tv_scale
        loss = losses.sum() * clip_guidance_scale + dir_loss
        return -torch.autograd.grad(loss, x)[0]

for i in range(n_batches):
    cur_t = diffusion.num_timesteps - skip_timesteps - 1

    samples = sample_fn(
        model,
        (batch_size, 3, model_config['image_size'], model_config['image_size']),
        clip_denoised=False,
        model_kwargs={'y': None},
        cond_fn=cond_fn,
        progress=True,
        skip_timesteps=skip_timesteps,
        init_image=init_image_tensor,
        randomize_class=False,
    )

    for j, sample in tqdm.tqdm(enumerate(samples)):
        cur_t -= 1
        if j % 100 == 0 or cur_t == -1:
            print()
            for k, image in enumerate(sample['pred_xstart']):
                filename = f'samples/progress_{i * batch_size + k:05}.png'
                TF.to_pil_image(image.add(1).div(2).clamp(0, 1)).save(filename)

