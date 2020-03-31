import argparse

from CompoundScalingSeg.dataset import NerveSegmentationDataset
from CompoundScalingSeg.transform import preprocessing
from CompoundScalingSeg.model import UNet, unet_params
from CompoundScalingSeg.trainer import Trainer
from CompoundScalingSeg.metric import dice_coeff

import torch
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size', type=int, default=5
)
parser.add_argument(
    '--epoch', type=int, default=20
)
parser.add_argument(
    '--lr', type=float, default=0.001
)
parser.add_argument(
    '--dataset', type=str, default='./data/'
)
parser.add_argument(
    '--workers', type=int, default=4
)
parser.add_argument(
    '--save_model', type=str, default='./save_model/'
)

cfg = parser.parse_args()
print(cfg)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == "__main__":
    ds_train = NerveSegmentationDataset(root='./data/', no_mask=False, train=True, transform=preprocessing, model_param='unet-w0')
    ds_test = NerveSegmentationDataset(root='./data/', no_mask=False, train=False, transform=preprocessing, model_param='unet-w0')
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)

    print("DATA LOADED")
    model = UNet(3, 1, True, 'unet-w0')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    success_metric = dice_coeff

    trainer = Trainer(model, criterion, optimizer, success_metric, device, None)
    fit = trainer.fit(dl_train, dl_test, num_epochs=cfg.epoch, checkpoints=cfg.save_model+model.__class__.__name__+'.pt')
    torch.save(model.state_dict(), './width_1.0/final_state_dict.pt')
    torch.save(model, './width_1.0/final.pt')

    loss_fn_name = "BCELoss"
    best_score = str(fit.best_score)
    print(f"Best loss score(loss function = {loss_fn_name}): {best_score}")