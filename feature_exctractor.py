import inspect
import torch

from torch import nn
from typing import List
from guided_diffusion.script_util import create_model_and_diffusion


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_tensors(module: nn.Module, features, name: str):
    """ Process and save activations in the module. """
    if type(features) in [list, tuple]:
        features = [f.detach().float() if f is not None else None 
                    for f in features]
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f.detach().float() for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.detach().float())


def save_out_hook(self, inp, out):
    save_tensors(self, out, 'activations')
    return out


def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], 'activations')
    return out


class FeatureExtractor(nn.Module):
    def __init__(self, model, input_activations: bool, **kwargs):
        ''' 
        Parent feature extractor class.
        
        param: model_path: path to the pretrained model
        param: input_activations: 
            If True, features are input activations of the corresponding blocks
            If False, features are output activations of the corresponding blocks
        '''
        super().__init__()
        self.model = model
        print(f"Pretrained model is successfully loaded")
        self.save_hook = save_input_hook if input_activations else save_out_hook
        self.feature_blocks = []


class FeatureExtractorDDPM(FeatureExtractor):
    ''' 
    Wrapper to extract features from pretrained DDPMs.
            
    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    '''
    
    def __init__(self, blocks: List[int], **kwargs):
        super().__init__(**kwargs)
        
        # Save decoder activations
        for idx, block in enumerate(self.model.input_blocks):
            if idx in blocks:
                block.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block)
    
    def get_activations(self):
        activations = []
    
        # Extract activations
        for block in self.feature_blocks:
                activations.append(block.activations)
                block.activations = None
        
        # Per-layer list of activations [N, C, H, W]
        return activations