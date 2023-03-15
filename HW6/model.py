import torch 
import torch.nn as nn

class SkipBlock(nn.Module):
    """
    This is a building-block class that I have used in several networks
    """
    def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
        super(SkipBlock, self).__init__()
        self.downsample = downsample
        self.skip_connections = skip_connections
        
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.convo1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        self.convo2 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if downsample:
            self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)
    def forward(self, x):
        identity = x                                     
        out = self.convo1(x)                              
        out = self.bn1(out)                              
        out = torch.nn.functional.relu(out)
        if self.in_ch == self.out_ch:
            out = self.convo2(out)                              
            out = self.bn2(out)                              
            out = torch.nn.functional.relu(out)
        if self.downsample:
            out = self.downsampler(out)
            identity = self.downsampler(identity)
        if self.skip_connections:
            if self.in_ch == self.out_ch:
                return out + identity                             
            else:
                out = torch.cat((out[:, :self.in_ch, :, :] + identity, out[:, self.in_ch:, :, :] + identity), dim=1)
                return out

class NetForYolo(nn.Module):
    """
    Recall that each YOLO vector is of size 5+C where C is the number of classes.  Since C
    equals 3 for the dataset used in the demo code in the Examples directory, our YOLO vectors
    are 8 elements long.  A YOLO tensor is a tensor representation of all the YOLO vectors
    created for a given training image.  The network shown below assumes that the input to
    the network is a flattened form of the YOLO tensor.  With an 8-element YOLO vector, a
    6x6 gridding of an image, and with 5 anchor boxes for each cell of the grid, the 
    flattened version of the YOLO tensor would be of size 1440.

    In Version 2.0.6 of the RPG module, I introduced a new loss function for this network
    that calls for using nn.CrossEntropyLoss for just the last C elements of each YOLO
    vector. [See Lines 64 through 83 of the code for "run_code_for_training_multi_instance_
    detection()" for how the loss is calculated in 2.0.6.]  Using nn.CrossEntropyLoss 
    required augmenting the last C elements of the YOLO vector with one additional 
    element for the purpose of representing the absence of an object in any given anchor
    box of a cell.  

    With the above mentioned augmentation, the flattened version of a YOLO tensor is
    of size 1620.  That is the reason for the one line change at the end of the 
    constructor initialization code shown below.
    """ 
    def __init__(self, skip_connections=True, depth=8):
        super(NetForYolo, self).__init__()
        # if depth not in [8,10,12,14,16]:
        #     sys.exit("This network has only been tested for 'depth' values 8, 10, 12, 14, and 16")
        self.depth = depth // 2
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1  = nn.BatchNorm2d(64)
        self.bn2  = nn.BatchNorm2d(128)

        self.skip64_arr = nn.ModuleList()
        for i in range(self.depth):
            self.skip64_arr.append(SkipBlock(64, 64, skip_connections=skip_connections))
        self.skip64ds =  SkipBlock(64,64, downsample=True, skip_connections=skip_connections)
        self.skip64to128 =  SkipBlock(64, 128, skip_connections=skip_connections )
        self.skip128_arr = nn.ModuleList()
        for i in range(self.depth):
            self.skip128_arr.append( SkipBlock(128,128, skip_connections=skip_connections))
        self.skip128ds =  SkipBlock(128,128, downsample=True, skip_connections=skip_connections)
        self.fc_seqn = nn.Sequential(
            nn.Linear(128*16*16, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 64*5*9),
            # nn.ReLU(inplace=True),
#                    nn.Linear(2048, 1440)
            # nn.Linear(2048, 1620)
        )

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))          
        x = nn.MaxPool2d(2,2)(torch.nn.functional.relu(self.conv2(x)))       
        for i,skip64 in enumerate(self.skip64_arr[:self.depth//4]):
            x = skip64(x)                
        x = self.skip64ds(x)
        for i,skip64 in enumerate(self.skip64_arr[self.depth//4:]):
            x = skip64(x)                
        x = self.bn1(x)
        x = self.skip64to128(x)
        for i,skip128 in enumerate(self.skip128_arr[:self.depth//4]):
            x = skip128(x)                
        x = self.bn2(x)
        x = self.skip128ds(x)
        # print("first:", x.shape)
        x = x.view(-1,128*16*16)
        # print("second:", x.shape)
        x = self.fc_seqn(x)
        # print("third", x.shape)
        return x