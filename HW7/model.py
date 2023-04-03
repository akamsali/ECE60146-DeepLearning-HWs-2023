import torch 
import torch.nn as nn


class DiscriminatorDG1(nn.Module):
    """
    This is an implementation of the DCGAN Discriminator. I refer to the DCGAN network topology as
    the 4-2-1 network.  Each layer of the Discriminator network carries out a strided
    convolution with a 4x4 kernel, a 2x2 stride and a 1x1 padding for all but the final
    layer. The output of the final convolutional layer is pushed through a sigmoid to yield
    a scalar value as the final output for each image in a batch.

    Class Path:  AdversarialLearning  ->   DataModeling  ->  DiscriminatorDG1
    """
    def __init__(self):
        super(DiscriminatorDG1, self).__init__()
        self.conv_in = nn.Conv2d(  3,    64,      kernel_size=4,      stride=2,    padding=1)
        self.conv_in2 = nn.Conv2d( 64,   128,     kernel_size=4,      stride=2,    padding=1)
        self.conv_in3 = nn.Conv2d( 128,  256,     kernel_size=4,      stride=2,    padding=1)
        self.conv_in4 = nn.Conv2d( 256,  512,     kernel_size=4,      stride=2,    padding=1)
        self.conv_in5 = nn.Conv2d( 512,  1,       kernel_size=4,      stride=1,    padding=0)
        self.bn1  = nn.BatchNorm2d(128)
        self.bn2  = nn.BatchNorm2d(256)
        self.bn3  = nn.BatchNorm2d(512)
        self.sig = nn.Sigmoid()
    def forward(self, x):                 
        x = torch.nn.functional.leaky_relu(self.conv_in(x), negative_slope=0.2, inplace=True)
        x = self.bn1(self.conv_in2(x))
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.bn2(self.conv_in3(x))
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.bn3(self.conv_in4(x))
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.conv_in5(x)
        x = self.sig(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.latent_to_image = nn.ConvTranspose2d( 100,   512,  kernel_size=4, stride=1, padding=0, bias=False)
        self.upsampler2 = nn.ConvTranspose2d( 512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampler3 = nn.ConvTranspose2d (256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampler4 = nn.ConvTranspose2d (128, 64,  kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampler5 = nn.ConvTranspose2d(  64,  3,  kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.tanh  = nn.Tanh()

    def forward(self, x):
        x = self.latent_to_image(x)
        x = torch.nn.functional.relu(self.bn1(x))
        x = self.upsampler2(x)
        x = torch.nn.functional.relu(self.bn2(x))
        x = self.upsampler3(x)
        x = torch.nn.functional.relu(self.bn3(x))
        x = self.upsampler4(x)
        x = torch.nn.functional.relu(self.bn4(x))
        x = self.upsampler5(x)
        x = self.tanh(x)
        return x
    

class SkipBlockDN(nn.Module):
    """
    This is a building-block class for constructing the Critic Network for adversarial learning.  In
    general, such a building-bloc class would be used for designing a network that creates a
    resolution hierarchy for the input image in which each successive layer is a downsampled version
    of the input image with or without increasing the number of input channels.

    Class Path:  AdversarialLearning  ->   DataModeling  ->  SkipBlockDN
    """
    def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
        super(SkipBlockDN, self).__init__()
        self.downsample = downsample
        self.skip_connections = skip_connections
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.convo1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.convo2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsampler1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2)
        self.downsampler2 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=2)
    def forward(self, x):
        identity = x                                     
        out = self.convo1(x)                              
        out = self.bn1(out)                              
        out = torch.nn.functional.relu(out)
        if self.in_ch == self.out_ch:
            out = self.convo2(out)                              
            out = self.bn2(out)                              
            out = torch.nn.functional.leaky_relu(out, negative_slope=0.2)
        if self.downsample:
            out = self.downsampler2(out)
            identity = self.downsampler1(identity)
        if self.skip_connections:
            out += identity                              
        return out

# class SkipBlockUP(nn.Module):
#     """
#     This is also a building-block class meant for a CNN that requires upsampling the images at the 
#     inputs to the successive layers.  I could use it in the Generator part of an Adversarial Network,
#     but have not yet done so.

#     Class Path:  AdversarialLearning  ->   DataModeling  ->  SkipBlockUP
#     """
#     def __init__(self, in_ch, out_ch, upsample=False, skip_connections=True):
#         super(SkipBlockUP, self).__init__()
#         self.upsample = upsample
#         self.skip_connections = skip_connections
#         self.in_ch = in_ch
#         self.out_ch = out_ch
#         self.convoT1 = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1)
#         self.convoT2 = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_ch)
#         self.bn2 = nn.BatchNorm2d(out_ch)
#         if upsample:
#             self.upsampler = nn.ConvTranspose2d(in_ch, out_ch, 1, stride=2, dilation=2, output_padding=1, padding=0)
#     def forward(self, x):
#         identity = x                                     
#         out = self.convoT1(x)                              
#         out = self.bn1(out)                              
#         out = torch.nn.functional.relu(out)
#         if self.in_ch == self.out_ch:
#             out = self.convoT2(out)                              
#             out = self.bn2(out)                              
#             out = torch.nn.functional.leaky_relu(out, negative_slope=0.2)
#         if self.upsample:
#             out = self.upsampler(out)
#             out = torch.nn.functional.leaky_relu(out, negative_slope=0.2)           
#             identity = self.upsampler(identity)
#             identity = torch.nn.functional.leaky_relu(identity, negative_slope=0.2) 
#         if self.skip_connections:
#             if self.in_ch == self.out_ch:
#                 out += identity                              
#             else:
#                 out += identity[:,self.out_ch:,:,:]
#             out = torch.nn.functional.leaky_relu(out, negative_slope=0.2)           
#         return out

class CriticCG1(nn.Module):
    """
    I have used the SkipBlockDN as a building block for the Critic network.  This I did with the hope
    that when time permits I may want to study the effect of skip connections on the behavior of the
    the critic vis-a-vis the Generator.  The final layer of the network is the same as in the 
    "official" GitHub implementation of Wasserstein GAN.  And, as in WGAN, I have used the leaky ReLU
    for activation.

    Class Path:  AdversarialLearning  ->   DataModeling  ->  CriticCG1
    """
    def __init__(self):
        super(CriticCG1, self).__init__()
        self.conv_in = SkipBlockDN(3, 64, downsample=True, skip_connections=True)
        self.conv_in2 = SkipBlockDN( 64,   128,  downsample=True, skip_connections=False)
        self.conv_in3 = SkipBlockDN(128,   256,  downsample=True, skip_connections=False)
        self.conv_in4 = SkipBlockDN(256,   512,  downsample=True, skip_connections=False)
        self.conv_in5 = SkipBlockDN(512,   1,  downsample=False, skip_connections=False)
        self.bn1  = nn.BatchNorm2d(128)
        self.bn2  = nn.BatchNorm2d(256)
        self.bn3  = nn.BatchNorm2d(512)
        self.final = nn.Linear(64, 1)
    def forward(self, x):              
        x = torch.nn.functional.leaky_relu(self.conv_in(x), negative_slope=0.2, inplace=True)
        # print("CriticCG1:  x.shape = ", x.shape)
        x = self.bn1(self.conv_in2(x))
        # print("CriticCG1:  x.shape = ", x.shape)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.bn2(self.conv_in3(x))
        # print("CriticCG1:  x.shape = ", x.shape)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.bn3(self.conv_in4(x))
        # print("CriticCG1:  x.shape = ", x.shape)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.conv_in5(x)
        # print("CriticCG1:  x.shape = ", x.shape)
        x = x.view(-1)
        # print("CriticCG1:  x.shape = ", x.shape)
        x = self.final(x)
        # print("CriticCG1:  x.shape = ", x.shape)
        # The following will cause a single value to be returned for the entire batch. This is
        # required by the Expectation operator E() in Equation (6) in the doc section of this 
        # file (See the beginning of this file).  For the P distribution in that equation, we 
        # apply the Critic directly to the training images.  And, for the Q distribution, we apply
        # the Critic to the output of the Generator. We need to use the expection operator for both.
        x = x.mean(0)       
        x = x.view(1)
        return x