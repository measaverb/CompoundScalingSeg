import random

import torch
import torch.nn as nn
import torchvision

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from CompoundScalingSeg.dataset import NerveSegmentationDataset
from CompoundScalingSeg.transform import preprocessing


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated(device=device)
    ngpus_per_node = torch.cuda.device_count()
    target_folder = './data/'

    ds_test = NerveSegmentationDataset(root=target_folder, train=False, transform=preprocessing, model_param='unet-w0')
    
    loc = './epoch_save_model/UNet_ckpt_19'
    model = torch.load(loc +'.pt')
    model = model.cuda()
    model.eval()

    pick = []
    for i in range(1):
        pick.append(random.randrange(0, 1000, 1))

    for i in pick:
        X, y = ds_test.__getitem__(i)
        torchvision.utils.save_image(X, './testimage/'+loc.split('/')[-1]+'_'+str(i)+'_X'+'.png')
        torchvision.utils.save_image(y, './testimage/'+loc.split('/')[-1]+'_'+str(i)+'_y'+'.png')
        X = X.view(1, 3, 300, 300).cuda()
        y_pred = model(X)
        torchvision.utils.save_image(y_pred, './testimage/'+loc.split('/')[-1]+'_'+str(i)+'_ypred'+'.png')
