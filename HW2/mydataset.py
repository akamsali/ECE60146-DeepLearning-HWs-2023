import glob
from PIL import Image
from torchvision import transforms as tvt
import torch


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root: 'str') -> None:
        super().__init__()
        # get file list with glob.glob
        self.file_list = sorted(glob.glob(root + '/*.jpg'))
        # set transform with tvt.Compose
        self.transform = tvt.Compose([tvt.Resize((256, 256)), tvt.ToTensor(), 
                                        tvt.Normalize(mean=[0.5, 0.5, 0.5], 
                                                    std=[0.5, 0.5, 0.5])])

    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, index: int):
        # read image with PIL.Image.open
        img = Image.open(self.file_list[index])
        # transform image with self.transform
        img = self.transform(img)
        label = self.file_list[index].split('/')[-1].split('.')[0]

        return img, int(label)