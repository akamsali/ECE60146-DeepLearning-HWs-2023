import torch.nn as nn

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