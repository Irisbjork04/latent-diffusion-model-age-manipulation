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
    config = OmegaConf.load("/work3/s212461/logs/2024-01-10T16-35-21_agemodel-ldm-v3/configs/2024-01-10T16-35-21-project.yaml")  
    model = load_model_from_config(config, "/work3/s212461/logs/2024-01-10T16-35-21_agemodel-ldm-v3/checkpoints/epoch=000078.ckpt")#"/work3/s212461/logs/2023-12-11T23-10-35_agemodel-ldm-v3/checkpoints/epoch=000028.ckpt")
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


def load_combined_embedding(embedding_path, age_label, image_name):
    # Load face embedding
    embedding_file = os.path.join(embedding_path, image_name.replace('.jpg', '.npy').replace('.png', '.npy'))
    face_embedding = torch.tensor(np.load(embedding_file)).float()

    # Convert age label to one-hot encoding
    one_hot_age = one_hot_encode_age(age_label, max_age=101)

    # Concatenate face embedding with one-hot encoded age
    combined_embedding = torch.cat((face_embedding, one_hot_age), dim=0)
    return combined_embedding

embeddings_path = "/work3/s212461/data/face_embeddings_train"


#ÍBS: I don't need this
#print("Create Morphing pairs for visualization")
#pairing_data = FRGCDataset("/mnt2/PhD-Marcel/ldm-face-manipulation/latent-diffusion/taming-transformers/data/morph_db")
#pair_indices = make_morph_pairs(pairing_data)

classes = list(range(50, 100, 30))


ages_for_sampling = list(range(1, 102))
#ages_for_sampling = (list(range(1, 101, 10)))

for i in classes:#tqdm(range(5)):
    x_stacked = []

    #for age in ages_for_sampling:

    image_name = ldm_data.datasets['train'][i]['image']  # Update this to get the correct image name
    x1 = torch.from_numpy(ldm_data.datasets['train'][i]['image'].reshape((1, 256, 256, 3))).to('cuda')

    face_embeddings = ldm_data.datasets['train'][i]['age'][101:].to('cuda')

    for age in ages_for_sampling:
        # Generate one-hot encoded age label for the current age
        age_label = one_hot_encode_age(age)

        face_embeddings_reshaped = face_embeddings.reshape((1, 1, -1))

        # Combine face embeddings with the generated age label
        combined_data = torch.cat((age_label, face_embeddings_reshaped), dim=2)

        print('ÍBS combined data for age {}: '.format(age), combined_data)
        


    # Combined face embeddings and age labels
    #combined_data = ldm_data.datasets['train'][i]['age'].to('cuda')
    #face_embeddings = combined_data[512:]
    #age_labels = combined_data[:101].reshape((1, 1, 101))

    # Separate the one-hot encoded age labels and face embeddings
    #age_labels = combined_data[:101].reshape((1, 1, 101))  # First 101 elements are age labels
    #face_embeddings = combined_data[101:]  # The rest are face embeddings


        print('ÍBS: Age labels:')
        print(age_label)
        print('ÍBS: Face Embeddings: ')
        print(face_embeddings)
    #x1 = torch.from_numpy(ldm_data.datasets['train'][i]['image'].reshape((1, 256, 256, 3))).to('cuda') #Images 
        #x2 = torch.from_numpy(ldm_data.datasets['train'][j]['image'].reshape((1, 256, 256, 3))).to("cuda")
        #c1 = one_hot_encode_age(age)
    #c1 = ldm_data.datasets['train'][i]['age'].reshape((1, 1, 101)).to('cuda')# Conditions

    # Reshape the combined data to the expected shape [1, 1, 613]
    # Adjust the shape based on your model's requirements
        c1 = combined_data.reshape((1, 1, -1))

    #print('ÍBS x1: ', x1)
        print('ÍBS combined data: ',c1)# c1)
        print(combined_data.shape)
        batch =  {'image': x1, 'age': age_label}
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
        index = torch.argmax(age_label).item()
        age_number = index
        Image.fromarray(denoise_grid.astype(np.uint8)).save(f"results/new-magface-{i}-age-{age_number}.png")


































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









