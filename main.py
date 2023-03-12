import torch
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

model_config = model_and_diffusion_defaults()
model_config.update({
    'attention_resolutions': '32, 16, 8',
    'class_cond': False,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing': '1000',  # Modify this value to decrease the number of
                                   # timesteps.
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model, diffusion = create_model_and_diffusion(**model_config)
model.load_state_dict(torch.load('unconditional_diffusion.pt', map_location='cpu'))
model.requires_grad_(False).eval().to(device)
