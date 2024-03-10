import sys
import os
import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from einops import rearrange
from omegaconf import OmegaConf

# Assuming paths and imports for your model setup are correct
sys.path.append('/work3/s212461')
sys.path.append("/data/Small_db")

from taming.models import vqgan
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def get_model():
    config_path = "/work3/s212461/logs/2024-01-10T16-35-21_agemodel-ldm-v3/configs/2024-01-10T16-35-21-project.yaml"
    ckpt_path = "/work3/s212461/logs/2024-01-10T16-35-21_agemodel-ldm-v3/checkpoints/epoch=000078.ckpt"
    config = OmegaConf.load(config_path)  
    model = load_model_from_config(config, ckpt_path)
    return model, config

model, config = get_model()
sampler = DDIMSampler(model)

print("Load and prepare Data")
ldm_data = instantiate_from_config(config.data)
ldm_data.prepare_data()
ldm_data.setup()

def one_hot_encode_age(age, max_age=101):
    encoding = torch.zeros((1, max_age)).to('cuda')  # Adjusted shape for simpler handling
    encoding[0, age - 1] = 1
    return encoding

def load_combined_embedding(embedding_path, age_label, image_name):
    embedding_file = os.path.join(embedding_path, image_name.replace('.jpg', '.npy').replace('.png', '.npy'))
    face_embedding = torch.tensor(np.load(embedding_file)).float().to('cuda')

    one_hot_age = one_hot_encode_age(age_label).unsqueeze(0)  # Adding batch dimension for consistency

    combined_embedding = torch.cat((one_hot_age, face_embedding.unsqueeze(0)), dim=1)  # Corrected concatenation
    return combined_embedding


image_path = "/work3/s212461/data/all_data_small_final_train"
def get_image_name_by_index(index):
    # List all jpg files in the directory
    image_files = [f for f in os.listdir(image_path) if f.endswith('.jpg')]
    image_files.sort()  # Ensure consistent order

    # Retrieve the image filename by index
    if index < len(image_files):
        return image_files[index]
    else:
        raise IndexError("Index out of range for the number of image files.")

embeddings_path = "/work3/s212461/data/face_embeddings_train"

classes = list(range(200, 300, 50))
ages_for_sampling = list(range(1, 102))

for i in classes:
    #image_name = ldm_data.datasets['train'][i]['image']  # Assuming this is how you get the image name
    #image_name = ldm_data.datasets['train'][i]['path']  # Adjusted to the correct key
    image_name = get_image_name_by_index(i) 
    x1 = torch.from_numpy(ldm_data.datasets['train'][i]['image'].reshape((1, 256, 256, 3))).to('cuda')

    for age in ages_for_sampling:
        combined_embedding = load_combined_embedding(embeddings_path, age, image_name)

        # Model inference with combined embedding
        # This section needs to be adjusted based on your model's specific inference process.
        # Assuming your model's inference function accepts combined embeddings and image tensors
        c1 = combined_embedding.reshape((1, 1, -1))
        print(f'Combined data for age {age}: ', c1.shape)

        # Here, you would call your model's inference function using `x1` and `c1`
        # e.g., output_image = model.inference(x1, c1)
        


    # Combined face embeddings and age labels
    #combined_data = ldm_data.datasets['train'][i]['age'].to('cuda')
    #face_embeddings = combined_data[512:]
    #age_labels = combined_data[:101].reshape((1, 1, 101))

    # Separate the one-hot encoded age labels and face embeddings
    #age_labels = combined_data[:101].reshape((1, 1, 101))  # First 101 elements are age labels
    #face_embeddings = combined_data[101:]  # The rest are face embeddings


       # print('ÍBS: Age labels:')
        #print(age_label)
        #print('ÍBS: Face Embeddings: ')
        #print(face_embeddings)
    #x1 = torch.from_numpy(ldm_data.datasets['train'][i]['image'].reshape((1, 256, 256, 3))).to('cuda') #Images 
        #x2 = torch.from_numpy(ldm_data.datasets['train'][j]['image'].reshape((1, 256, 256, 3))).to("cuda")
        #c1 = one_hot_encode_age(age)
    #c1 = ldm_data.datasets['train'][i]['age'].reshape((1, 1, 101)).to('cuda')# Conditions

    # Reshape the combined data to the expected shape [1, 1, 613]
    # Adjust the shape based on your model's requirements
       # c1 = combined_data.reshape((1, 1, -1))

    #print('ÍBS x1: ', x1)
       # print('ÍBS combined data: ',c1)# c1)
       # print(combined_data.shape)
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
        #index = torch.argmax(age).item()
        age_number = age
        Image.fromarray(denoise_grid.astype(np.uint8)).save(f"results/openAI-magface_onehot-{i}-age-{age_number}.png")


































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









