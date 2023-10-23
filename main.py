import clip
import torch
import tqdm

from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models
from torchvision.transforms import functional as TF

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from feature_exctractor import FeatureExtractorDDPM

model_config = model_and_diffusion_defaults()
model_config.update({
    'attention_resolutions': '32, 16, 8',
    'class_cond': False,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing': '50', # see sampling scheme in 4.1 (T')
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
model.load_state_dict(torch.load('models/unconditional_diffusion.pt', map_location='cpu'))
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

VGG = models.vgg19(pretrained=True).features
VGG.to(device)
for parameter in VGG.parameters():
    parameter.requires_grad_(False)

# Define loss-related functions

def global_loss(image, prompt):
    similarity = 1 - clip_model(image, prompt)[0] / 100 # clip returns the cosine similarity times 100
    return similarity.mean()

def directional_loss(x, x_t, p_source, p_target):
    encoded_image_diff = x - x_t
    encoded_text_diff = p_source - p_target
    cosine_similarity = torch.nn.functional.cosine_similarity(
        encoded_image_diff,
        encoded_text_diff,
        dim=-1
    )
    return (1 - cosine_similarity).mean()

def zecon_loss(x0_features_list, x0_t_features_list, temperature=0.07):
    loss_sum = 0
    num_layers = len(x0_features_list)

    for x0_features, x0_t_features in zip(x0_features_list, x0_t_features_list):
        batch_size, feature_dim, h, w = x0_features.size()
        x0_features = x0_features.view(batch_size, feature_dim, -1)
        x0_t_features = x0_t_features.view(batch_size, feature_dim, -1)

        # Compute the similarity matrix
        sim_matrix = torch.einsum('bci,bcj->bij', x0_features, x0_t_features)
        sim_matrix = sim_matrix / temperature

        # Create positive and negative masks
        pos_mask = torch.eye(h * w, device=sim_matrix.device).unsqueeze(0).bool()
        neg_mask = ~pos_mask

        # Compute the loss using cross-entropy
        logits = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0]
        labels = torch.arange(h * w, device=logits.device)
        logits_1d = logits.view(-1)[neg_mask.view(-1)]
        labels_1d = labels.repeat(batch_size * (h * w - 1)).unsqueeze(0).to(torch.float)
        layer_loss = F.cross_entropy(logits_1d.view(batch_size, -1), labels_1d, reduction='mean')

        loss_sum += layer_loss

    # Average the loss across layers
    loss = loss_sum / num_layers

    return loss

def get_features(image, model, layers=None):

    if layers is None:
        layers = {'0': 'conv1_1',  
                  '5': 'conv2_1',  
                  '10': 'conv3_1', 
                  '19': 'conv4_1', 
                  '21': 'conv4_2', 
                  '28': 'conv5_1',
                  '31': 'conv5_2'
                 }  
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)   
        if name in layers:
            features[layers[name]] = x
    
    return features

def feature_loss(x, x_t):
    x_features = get_features(x, VGG)
    x_t_features = get_features(x_t, VGG)

    loss = 0
    loss += torch.mean((x_features['conv4_2'] - x_t_features['conv4_2']) ** 2)
    loss += torch.mean((x_features['conv5_2'] - x_t_features['conv5_2']) ** 2)

    return loss

def pixel_loss(x, x_t):
    loss = nn.MSELoss()
    return loss(x, x_t)

# Run clip-guided diffusion

p_source = "portrait"
p_target = "3d render in the style of Pixar"
batch_size = 1
clip_guidance_scale = 1
skip_timesteps = 25 # see sampling scheme in 4.1 (t0)
cutn = 32
cut_pow = 0.5
n_batches = 1
seed = 17

if seed is not None:
    torch.manual_seed(seed)

text_embed_source = clip_model.encode_text(clip.tokenize(p_source).to(device)).float()
text_embed_target = clip_model.encode_text(clip.tokenize(p_target).to(device)).float()
text_target_tokens = clip.tokenize(p_target).to(device)

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

# Patcher

resize_cropper = transforms.RandomResizedCrop(size=(clip_size, clip_size))
affine_transfomer = transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
perspective_transformer = transforms.RandomPerspective(distortion_scale=0.6, p=1.0)
patcher = transforms.Compose([
    resize_cropper,
    perspective_transformer,
    affine_transfomer
])

def img_normalize(image):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
    std=torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

# Feature Exctractor
feature_extractor = FeatureExtractorDDPM(
    model = model,
    blocks = [10, 11, 12, 13, 14],
    input_activations = False,
    **model_config
)

# Conditioning function

def cond_fn(x, t, y=None):
    with torch.enable_grad():
        x = x.detach().requires_grad_()
        n = x.shape[0]
        my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
        out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
        fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
        x_in = out['pred_xstart'] * fac + x * (1 - fac)
        x_in_patches = torch.cat([normalize(patcher(x_in.add(1).div(2))) for i in range(cutn)])
        x_in_patches_embeddings = clip_model.encode_image(x_in_patches).float()
        g_loss = global_loss(x_in_patches, text_target_tokens)
        dir_loss = directional_loss(init_image_embedding, x_in_patches_embeddings, text_embed_source, text_embed_target)
        feat_loss = feature_loss(img_normalize(init_image_tensor), img_normalize(x_in))
        mse_loss = pixel_loss(init_image_tensor, x_in)
        x_t_features = feature_extractor.get_activations() # unet features
        model(init_image_tensor, t)
        x_0_features = feature_extractor.get_activations() # unet features
        z_loss = zecon_loss(x_0_features, x_t_features)

        loss = g_loss * 5000 + dir_loss * 5000 + feat_loss * 100 + mse_loss * 10000 + z_loss * 500
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
        if j % 5 == 0 or cur_t == -1:
            # print()
            for k, image in enumerate(sample['pred_xstart']):
                filename = f'samples/progress_{i * batch_size + k:05}.png'
                TF.to_pil_image(image.add(1).div(2).clamp(0, 1)).save(filename)

