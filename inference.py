import sys
#sys.path.append(".")
sys.path.append('/work3/s212461')
sys.path.append("/data/Small_db")
from taming.models import vqgan
from ldm.models.diffusion.ddim import DDIMSampler 
from einops import rearrange
from torchvision.utils import make_grid
from torchvision import transforms
from ldm.util import instantiate_from_config, default
from omegaconf import OmegaConf
#from frgc import FRGCDataset
#from pair_selection import make_morph_pairs
#from utils.visual import draw_bbox
#from utils.file import get_image
from PIL import Image
from tqdm import tqdm
import os
import torch
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#map_location=torch.device('cpu')

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location=torch.device("cpu"))
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("/work3/s212461/logs/2023-12-11T23-10-35_agemodel-ldm-v3/configs/2023-12-11T23-10-35-project.yaml")  
    model = load_model_from_config(config, "/work3/s212461/logs/2023-12-11T23-10-35_agemodel-ldm-v3/checkpoints/epoch=000030.ckpt")
    return model, config


print("Load Latent Diffusion Model")
model, config = get_model()
sampler = DDIMSampler(model)

print("Load and prepare Data data")
ldm_data = instantiate_from_config(config.data)
ldm_data.prepare_data()
ldm_data.setup()

#Function to transfer age to one hot encoded tensors
def one_hot_encode_age(age, max_age=101):
    # Create a zero tensor of shape [1, 1, max_age]
    encoding = torch.zeros((1, 1, max_age)).to('cuda')
    # Set the index corresponding to the age to 1
    encoding[0, 0, age - 1] = 1
    return encoding

#ÍBS: I don't need this
#print("Create Morphing pairs for visualization")
#pairing_data = FRGCDataset("/mnt2/PhD-Marcel/ldm-face-manipulation/latent-diffusion/taming-transformers/data/morph_db")
#pair_indices = make_morph_pairs(pairing_data)

classes = list(range(1, 10000, 10))

ages_for_sampling = list(range(1, 101))
#ages_for_sampling = (list(range(1, 101, 10)))

for i in classes:#tqdm(range(5)):
    x_stacked = []

    for age in ages_for_sampling:

        x1 = torch.from_numpy(ldm_data.datasets['train'][i]['image'].reshape((1, 256, 256, 3))).to('cuda') #Images 
        
        #x2 = torch.from_numpy(ldm_data.datasets['train'][j]['image'].reshape((1, 256, 256, 3))).to("cuda")
        c1 = one_hot_encode_age(age)
        c2 = ldm_data.datasets['train'][i]['age'].reshape((1, 1, 101)).to('cuda')# Conditions

    #print('ÍBS x1: ', x1)
    #print('ÍBS c1: ', c1)
        batch =  {'image': x1, 'age': c1}
        batch['age'] = batch['age'].squeeze(1)
        N=1
        print("Batch shape before model.get_input:", batch['image'].shape)
        z, c, x, xrec, xc = model.get_input(batch, 'image',
                                    return_first_stage_outputs=True,
                                    force_c_encode=True,
                                    return_original_cond=True,
                                    bs=N)

        ts = torch.full((1,), 999, device=model.device, dtype=torch.long)

        #random_noise = torch.randn_like(z)
        z_t = model.q_sample(x_start = z, t = ts, noise = None)

        #z_t = model.q_sample(x_start = z, t = ts, noise = None)

        #img, progressives = model.progressive_denoising(cm, shape=(3, 64, 64), batch_size=1, x_T = z_t, start_T=999, x0 = z)

        #img, progressives = model.progressive_denoising(c1, shape=(3, 64, 64), batch_size=1, x_T = z_t, start_T=999, x0 = z) #this img is the Z
        #try:
        img, progressives = model.progressive_denoising(c1, shape=(3, 64, 64), batch_size=1, start_T=999, x0 = z) #x_T=z_t,
        #except Exception as e:
        #    print(f"Error during sampling: {e}")
        x_morphed = model.decode_first_stage(img) # the actual image
        #Rest is for visualization
        x_morphed = rearrange(x_morphed, 'b c h w -> b h w c')
        x_stacked = torch.stack([x1, x_morphed]).squeeze()
        x_stacked = (x_stacked + 1.0) / 2.0
        denoise_grid = rearrange(x_stacked, 'b h w c -> b c h w')
        denoise_grid = 255. * make_grid(denoise_grid, nrow=1).cpu().numpy()
        denoise_grid = rearrange(denoise_grid, 'c h w -> h w c')


        #Convert age:
        index = torch.argmax(c1).item()
        age_number = index+1
        index2 = torch.argmax(c2).item()
        age_number2 = index2+1
        Image.fromarray(denoise_grid.astype(np.uint8)).save(f"results_onehot/OneHotV5-{i}-age-{age_number}_from_age-{age_number2}.png")


































"""

BACKLOG CODE

batch =  {'image': x, 'fr_embeds': cm}

z, c, x, xrec, xc = model.get_input(batch, 'image',
                                    return_first_stage_outputs=True,
                                    force_c_encode=True,
                                    return_original_cond=True,
                                    bs=N)

ts = torch.full((1,), 999, device=model.device, dtype=torch.long)
z_t = model.q_sample(x_start = z, t = ts, noise = None)

#img, progressives = model.progressive_denoising(cm, shape=(3, 64, 64), batch_size=1, x_T = z_t, start_T=999, x0 = z)
img, progressives = model.progressive_denoising(cm, shape=(3, 64, 64), batch_size=1, start_T=999)
#prog_row = model._get_denoise_row_from_list(progressives, desc="Progressive Generation")
#prog_row = rearrange(prog_row, 'c h w -> h w c').cpu().numpy()
x_samples = model.decode_first_stage(img)
# prog_row = (255 * (prog_row + 1.0) / 2.0)
# Image.fromarray(prog_row.astype(np.uint8)).save("magface-test.png")

x_samples = x_samples.squeeze()
x_samples = (x_samples + 1.0) / 2.0
x_samples = 255. * rearrange(x_samples, 'c h w -> h w c').cpu().numpy()

Image.fromarray(x_samples.astype(np.uint8)).save("magface-test-1.png")
"""









