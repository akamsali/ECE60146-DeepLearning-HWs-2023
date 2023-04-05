import model

import torch
import torch.nn as nn
from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3

import pickle 
import numpy as np

def weights_init(m):        
    """
    Uses the DCGAN initializations for the weights
    """
    classname = m.__class__.__name__     
    if classname.find('Conv') != -1:         
        nn.init.normal_(m.weight.data, 0.0, 0.02)      
    elif classname.find('BatchNorm') != -1:         
        nn.init.normal_(m.weight.data, 1.0, 0.02)       
        nn.init.constant_(m.bias.data, 0)    


def generate_fake_images(model_path, device, num_imgs=1000):
    generator = model.Generator()
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    # for i in range(1000):
    noise = torch.randn(num_imgs, 100, 1, 1, device=device, dtype=torch.float)
    fake = generator(noise)
    fake = fake.detach().cpu().numpy()
    fake = ((fake + 1) * 127.5).astype(np.uint8)

    with open(f'./solutions/{model_path}_fake_data.pkl', 'wb') as f:
        pickle.dump(fake, f)


def get_FID_score(real_image_files, fake_image_files):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx]).to(device)

    m1, s1 = calculate_activation_statistics(real_image_files, 
                                            inception_model,
                                            device=device)
    m2, s2 = calculate_activation_statistics(fake_image_files,
                                                inception_model,
                                                device=device)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    print(f"FID score: {fid_value}")