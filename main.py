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

        if not os.path.exists(self.path):
            raise Exception(f"[!] dataset is not exited")

        self.image_file_name = sorted(
            np.random.choice(
                os.listdir(os.path.join(self.path, "hr")),
                size=num_images,
                replace=False,
            )
        )
        # self.mean = np.array([0.485, 0.456, 0.406])
        # self.std = np.array([0.229, 0.224, 0.225])
        self.mean = np.array([0.5, 0.5, 0.5])
        self.std = np.array([0.5, 0.5, 0.5])

    def __getitem__(self, item):
        file_name = self.image_file_name[item]
        high_resolution = Image.open(os.path.join(self.path, "hr", file_name)).convert(
            "RGB"
        )
        low_resolution = Image.open(os.path.join(self.path, "lr", file_name)).convert(
            "RGB"
        )

        if train:
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

        # high_resolution = TF.normalize(high_resolution, self.mean, self.std)
        # high_resolution = TF.normalize(high_resolution, self.mean, self.std)

        # high_resolution = TF.normalize(high_resolution, self.mean, self.std)
        # low_resolution = TF.normalize(low_resolution, self.mean, self.std)

        images = {"lr": low_resolution, "hr": high_resolution}

        return images

    def __len__(self):
        return len(self.image_file_name)


config = {
    "image_size": 512,
    "batch_size": 12,
    "start_epoch": 0,
    "num_epoch": 20,
    "sample_batch_size": 1,
    "checkpoint_dir": "./checkpoints",
    "sample_dir": "./samples",
    "workers": 6,
    "scale_factor": 4,
    "num_rrdn_blocks": 11,
    "nf": 32,
    "gc": 32,
    "b1": 0.9,
    "b2": 0.999,
    "weight_decay": 1e-2,
    # ------ PSNR ------
    "p_lr": 2e-4,
    "p_decay_iter": [4, 8, 12, 16],
    "p_perceptual_loss_factor": 0,
    "p_adversarial_loss_factor": 0,
    "p_content_loss_factor": 1,
    # ------------------
    # ------ ADVR ------
    "g_lr": 1e-4,
    "g_decay_iter": [10, 20, 35, 50],
    "g_perceptual_loss_factor": 1,
    "g_adversarial_loss_factor": 5e-3,
    "g_content_loss_factor": 1e-2,
    # ------------------
    "is_psnr_oriented": True,
    "load_previous_opt": True,
}

import train

importlib.reload(train)

if not os.path.exists(config["sample_dir"]):
    os.makedirs(config["sample_dir"])


pin = torch.cuda.is_available()


config["start_epoch"] = 2
config["num_epoch"] = 10
config["batch_size"] = 12
config["is_psnr_oriented"] = True
# for loading last ran checkpoint optimizers parameters
config["load_previous_opt"] = True



esr_dataset_train = ESR_Dataset(num_images=9000, path=r"./images", train=True)
esr_dataloader_train = DataLoader(
    esr_dataset_train,
    config["batch_size"],
    num_workers=config["workers"],
    pin_memory=pin,
    shuffle=True,
)

esr_dataset_val = ESR_Dataset(num_images=config["batch_size"], path=r"./images", train=False)
esr_dataloader_val = DataLoader(
    esr_dataset_val, config["batch_size"], num_workers=config["workers"], pin_memory=pin,
)

for key, value in config.items():
    print(f"{key:30}: {value}")

print("\n\n\n")
print(f"ESRGAN start")

torch.cuda.empty_cache()
trainer = train.Trainer(config, esr_dataloader_train, esr_dataloader_val, device)
train_metrics = trainer.train()
