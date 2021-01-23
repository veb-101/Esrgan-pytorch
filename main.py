import train
import os
import random
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import importlib


seed = 41
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


device = get_default_device()


class ESR_Dataset(Dataset):
    def __init__(self, num_images=9000, path=r"images", train=True):
        self.path = path
        self.is_train = train

        if not os.path.exists(self.path):
            raise Exception(f"[!] dataset is not exited")

        self.image_paths = os.listdir(os.path.join(self.path, "hr"))
        # self.image_range = image_range if image_range else (0, len(self.image_paths))
        # if len(self.image_range) == 1:
        #     if self.is_train:
        #         self.image_range = (0, self.image_range[0])
        #     else:
        #         self.image_range = (self.image_range[0], len(self.image_paths))

        # self.start = self.image_range[0]
        # self.end = self.image_range[1]

        self.image_file_name = np.random.choice(self.image_paths, size=num_images, replace=False)

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        # self.mean = np.array([0.5, 0.5, 0.5])
        # self.std = np.array([0.5, 0.5, 0.5])

    def __getitem__(self, item):
        file_name = self.image_file_name[item]
        high_resolution = Image.open(os.path.join(self.path, "hr", file_name)).convert(
            "RGB"
        )
        low_resolution = Image.open(os.path.join(self.path, "lr", file_name)).convert(
            "RGB"
        )

        if self.is_train:
            if random.random() > 0.5:
                high_resolution = TF.vflip(high_resolution)
                low_resolution = TF.vflip(low_resolution)

            if random.random() > 0.5:
                high_resolution = TF.hflip(high_resolution)
                low_resolution = TF.hflip(low_resolution)

            if random.random() > 0.5:
                high_resolution = TF.rotate(high_resolution, 90)
                low_resolution = TF.rotate(low_resolution, 90)

        high_resolution = TF.to_tensor(high_resolution)
        low_resolution = TF.to_tensor(low_resolution)

        high_resolution = TF.normalize(high_resolution, self.mean, self.std)
        low_resolution = TF.normalize(low_resolution, self.mean, self.std)

        images = {"lr": low_resolution, "hr": high_resolution}

        return images

    def __len__(self):
        return len(self.image_file_name)


config = {
    "image_size": 256,
    "batch_size": 16,
    "start_epoch": 0,
    "num_epoch": 100,
    "sample_batch_size": 1,
    "sample_dir": "./samples",
    "workers": 6,
    "scale_factor": 4,
    "num_rrdn_blocks": 18,
    "nf": 64,
    "gc": 32,
    "b1": 0.9,
    "b2": 0.999,
    "weight_decay": 1e-2,
    # ------ PSNR ------
    "p_lr": 2e-4,
    "p_decay_iter": [20, 40, 60, 80],
    "p_perceptual_loss_factor": 0,
    "p_adversarial_loss_factor": 0,
    "p_content_loss_factor": 1,
    # ------------------
    # ------ ADVR ------
    "g_lr": 1e-4,
    "g_decay_iter": [20, 40, 60, 80],
    "g_perceptual_loss_factor": 1,
    "g_adversarial_loss_factor": 5e-3,
    "g_content_loss_factor": 1e-2,
    # ------------------
    "is_psnr_oriented": True,
    "load_previous_opt": True,
}


importlib.reload(train)

if not os.path.exists(config["sample_dir"]):
    os.makedirs(config["sample_dir"])


pin = torch.cuda.is_available()


config["start_epoch"] = 0
config["num_epoch"] = 100
config["batch_size"] = 16
config["is_psnr_oriented"] = True
config["load_previous_opt"] = True


esr_dataset_train = ESR_Dataset(
    num_images=9000, path=r"./images/train", train=True)

esr_dataset_val = ESR_Dataset(
    num_images=64, path=r"./images/valid", train=False)

esr_dataloader_train = DataLoader(
    esr_dataset_train,
    config["batch_size"],
    num_workers=config["workers"],
    pin_memory=pin,
    shuffle=True,
)

esr_dataloader_val = DataLoader(
    esr_dataset_val,
    config["batch_size"],
    num_workers=config["workers"],
    pin_memory=pin,
)

# for key, value in config.items():
#     print(f"{key:30}: {value}")

print("\n\n\n")
print(f"ESRGAN start")

# torch.cuda.empty_cache()
# trainer = train.Trainer(config, esr_dataloader_train, esr_dataloader_val, device)
# train_metrics = trainer.train()
