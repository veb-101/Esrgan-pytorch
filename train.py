import os
from glob import glob
import numpy as np
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.autograd import Variable
from torchvision.utils import save_image
from models import PerceptualLoss, Generator, Discriminator
import shutil
import cv2
from utils import cal_img_metrics
import gc
from torch.cuda import amp


class Trainer:
    def __init__(self, config, data_loader_train, data_loader_val, device):
        self.device = device
        self.num_epoch = config["num_epoch"]
        self.start_epoch = config["start_epoch"]
        self.image_size = config["image_size"]
        self.checkpoint_dir = config["checkpoint_dir"]
        self.sample_dir = config["sample_dir"]

        self.batch_size = config["batch_size"]
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        self.num_res_blocks = config["num_rrdn_blocks"]
        self.nf = config["nf"]
        self.scale_factor = config["scale_factor"]
        self.is_psnr_oriented = config["is_psnr_oriented"]
        self.load_previous_opt = config["load_previous_opt"]

        if self.is_psnr_oriented:
            self.lr = config["p_lr"]
            self.content_loss_factor = config["p_content_loss_factor"]
            self.perceptual_loss_factor = config["p_perceptual_loss_factor"]
            self.adversarial_loss_factor = config["p_adversarial_loss_factor"]
            self.decay_iter = config["p_decay_iter"]
        else:
            self.lr = config["g_lr"]
            self.content_loss_factor = config["g_content_loss_factor"]
            self.perceptual_loss_factor = config["g_perceptual_loss_factor"]
            self.adversarial_loss_factor = config["g_adversarial_loss_factor"]
            self.decay_iter = config["g_decay_iter"]

        self.metrics = {
            "dis_loss": [],
            "gen_loss": [],
            "per_loss": [],
            "con_loss": [],
            "adv_loss": [],
            "SSIM": [],  # validation set per epoch
            "PSNR": [],  # validation set per epoch
        }

        self.build_model(config)
        self.lr_scheduler_generator = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer_generator, self.decay_iter
        )
        self.lr_scheduler_discriminator = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer_discriminator, self.decay_iter
        )

    def train(self):
        os.makedirs("/content/drive/MyDrive/Project-ESRGAN", exist_ok=True)

        total_step = len(self.data_loader_train)
        adversarial_criterion = nn.BCEWithLogitsLoss().to(self.device)
        content_criterion = nn.L1Loss().to(self.device)
        perception_criterion = PerceptualLoss().to(self.device)

        self.generator.train()
        self.discriminator.train()

        steps_completed = 0
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        for epoch in range(self.start_epoch, self.start_epoch + self.num_epoch):

            steps_completed = (self.start_epoch + 1) * total_step

            if not os.path.exists(os.path.join(self.sample_dir, str(epoch))):
                os.makedirs(os.path.join(self.sample_dir, str(epoch)))

            for step, image in enumerate(self.data_loader_train):

                # print("step", step)
                low_resolution = image["lr"].to(self.device)
                high_resolution = image["hr"].to(self.device)

                # Adversarial ground truths
                real_labels = Variable(
                    Tensor(
                        np.ones(
                            (low_resolution.size(0), *self.discriminator.output_shape)
                        )
                    ),
                    requires_grad=False,
                )
                fake_labels = Variable(
                    Tensor(
                        np.zeros(
                            (low_resolution.size(0), *self.discriminator.output_shape)
                        )
                    ),
                    requires_grad=False,
                )

                ##########################
                #   training generator   #
                ##########################
                self.optimizer_generator.zero_grad()

                with amp.autocast():

                    fake_high_resolution = self.generator(low_resolution)

                    # Content loss - L1 loss - psnr oriented
                    content_loss = content_criterion(
                        fake_high_resolution, high_resolution
                    )

                    if not self.is_psnr_oriented:
                        score_real = self.discriminator(high_resolution)
                        score_fake = self.discriminator(fake_high_resolution)

                        # ----------------------
                        # calculate Realtivistic GAN loss Drf and Dfr
                        discriminator_rf = score_real - score_fake.mean(0, keepdim=True)
                        discriminator_fr = score_fake - score_real.mean(0, keepdim=True)

                        adversarial_loss_rf = adversarial_criterion(
                            discriminator_rf, fake_labels
                        )
                        adversarial_loss_fr = adversarial_criterion(
                            discriminator_fr, real_labels
                        )
                        adversarial_loss = (
                            adversarial_loss_fr + adversarial_loss_rf
                        ) / 2
                        # ----------------------

                        # Perceptual loss - VGG loss before activations
                        perceptual_loss = perception_criterion(
                            high_resolution, fake_high_resolution
                        )

                        generator_loss = (
                            adversarial_loss * self.adversarial_loss_factor
                            + perceptual_loss * self.perceptual_loss_factor
                            + content_loss * self.content_loss_factor
                        )

                    else:
                        generator_loss = content_loss * self.content_loss_factor

                self.scaler_gen.scale(generator_loss).backward()
                self.scaler_gen.step(self.optimizer_generator)
                self.scaler_gen.update()
                # self.optimizer_generator.step()

                self.metrics["gen_loss"].append(
                    np.round(generator_loss.detach().cpu().item(), 5)
                )
                self.metrics["con_loss"].append(
                    np.round(
                        content_loss.detach().cpu().item() * self.content_loss_factor, 4
                    )
                )
                torch.cuda.empty_cache()
                gc.collect()

                ##########################
                # training discriminator #
                ##########################
                if not self.is_psnr_oriented:
                    self.optimizer_discriminator.zero_grad()

                    with amp.autocast():
                        score_real = self.discriminator(high_resolution)
                        score_fake = self.discriminator(fake_high_resolution.detach())
                        discriminator_rf = score_real - score_fake.mean(
                            axis=0, keepdim=True
                        )
                        discriminator_fr = score_fake - score_real.mean(
                            axis=0, keepdim=True
                        )

                        adversarial_loss_rf = adversarial_criterion(
                            discriminator_rf, real_labels
                        )
                        adversarial_loss_fr = adversarial_criterion(
                            discriminator_fr, fake_labels
                        )
                        discriminator_loss = (
                            adversarial_loss_fr + adversarial_loss_rf
                        ) / 2

                    self.scaler_dis.scale(discriminator_loss).backward()
                    self.scaler_dis.step(self.optimizer_discriminator)
                    self.scaler_dis.update()
                    # discriminator_loss.backward()
                    # self.optimizer_discriminator.step()

                    self.metrics["dis_loss"].append(
                        np.round(discriminator_loss.cpu().detach().item(), 5)
                    )

                    # generator metrics
                    self.metrics["adv_loss"].append(
                        np.round(
                            adversarial_loss.detach().cpu().item()
                            * self.adversarial_loss_factor,
                            4,
                        )
                    )
                    self.metrics["per_loss"].append(
                        np.round(
                            perceptual_loss.detach().cpu().item()
                            * self.perceptual_loss_factor,
                            4,
                        )
                    )

                if step == int(total_step / 2) or step == 0 or step == (total_step - 1):
                    if not self.is_psnr_oriented:
                        print(
                            f"[Epoch {epoch}/{self.start_epoch+self.num_epoch-1}] [Batch {step+1}/{total_step}]"
                            f"[D loss {self.metrics['dis_loss'][-1]}] [G loss {self.metrics['gen_loss'][-1]}]"
                            f"[adversarial loss {self.metrics['adv_loss'][-1]}]"
                            f"[perceptual loss {self.metrics['per_loss'][-1]}]"
                            f"[content loss {self.metrics['con_loss'][-1]}]"
                            f""
                        )
                    else:
                        print(
                            f"[Epoch {epoch}/{self.start_epoch+self.num_epoch-1}] [Batch {step+1}/{total_step}] "
                            f"[content loss {self.metrics['con_loss'][-1]}]"
                        )

                    result = torch.cat(
                        (
                            high_resolution.cpu().detach(),
                            fake_high_resolution.cpu().detach(),
                        ),
                        2,
                    )
                    # print(f"result:", result[0].min(), result[0].max())
                    save_image(
                        result.clamp(0.0, 1.0),
                        os.path.join(self.sample_dir, str(epoch), f"ESR_{step}.png"),
                    )
                torch.cuda.empty_cache()
                gc.collect()

            # validation set SSIM and PSNR
            for image_val in self.data_loader_val:
                val_low_resolution = image_val["lr"].to(self.device)
                val_high_resolution = image_val["hr"].to(self.device)
                with torch.no_grad():
                    with amp.autocast():
                        val_fake_high_res = self.generator(val_low_resolution)

                    # image metrics PSNR and SSIM
                    val_psnr, val_ssim = cal_img_metrics(
                        val_fake_high_res.cpu().detach(),
                        val_high_resolution.cpu().detach(),
                    )
                    self.metrics["PSNR"].append(val_psnr)
                    self.metrics["SSIM"].append(val_ssim)

                    result_val = torch.cat(
                        (
                            val_high_resolution.cpu().detach(),
                            val_fake_high_res.cpu().detach(),
                        ),
                        2,
                    )
                    # print(f"result_val:", result_val[0].min(), result_val[0].max())
                    save_image(
                        result_val.clamp(0.0, 1.0),
                        os.path.join(self.sample_dir, f"Validation_{epoch}.png"),
                    )

                print(f"Validation Set: PSNR: {val_psnr}, SSIM:{val_ssim}")

            self.lr_scheduler_generator.step()
            if not self.is_psnr_oriented:
                self.lr_scheduler_discriminator.step()

            torch.save(
                {
                    "next_epoch": epoch + 1,
                    f"generator_dict_{epoch}": self.generator.state_dict(),
                    f"discriminator_dict_{epoch}": self.discriminator.state_dict(),
                    f"optim_gen_{epoch}": self.optimizer_generator.state_dict(),
                    f"optim_dis_{epoch}": self.optimizer_discriminator.state_dict(),
                    f"steps_completed": steps_completed,
                    f"metrics_till_{epoch}": self.metrics,
                    f"grad_scaler_gen_{epoch}": self.scaler_gen,
                    f"grad_scaler_dis_{epoch}": self.scaler_dis,
                },
                f"checkpoint_{epoch}.tar",
            )
            shutil.copyfile(
                f"checkpoint_{epoch}.tar",
                os.path.join(
                    r"/content/drive/MyDrive/Project-ESRGAN", rf"checkpoint_{epoch}.tar"
                ),
            )
            torch.cuda.empty_cache()
            gc.collect()
        return self.metrics

    def build_model(self, config):

        self.generator = Generator(
            channels=3,
            nf=self.nf,
            num_res_blocks=self.num_res_blocks,
            scale=self.scale_factor,
        ).to(self.device)

        self.generator._mrsa_init(self.generator.layers_)

        self.discriminator = Discriminator(
            input_shape=(3, self.image_size, self.image_size)
        ).to(self.device)

        self.optimizer_generator = Adam(
            self.generator.parameters(),
            lr=self.lr,
            betas=(config["b1"], config["b2"]),
            weight_decay=config["weight_decay"],
        )
        self.optimizer_discriminator = Adam(
            self.discriminator.parameters(),
            lr=self.lr,
            betas=(config["b1"], config["b2"]),
            weight_decay=config["weight_decay"],
        )

        self.scaler_gen = torch.cuda.amp.GradScaler()
        self.scaler_dis = torch.cuda.amp.GradScaler()

        self.load_model()

    def load_model(self,):
        drive_path = r"/content/drive/MyDrive/Project-ESRGAN"
        print(f"[*] Finding checkpoint {self.start_epoch-1} in {drive_path}")

        checkpoint_file = f"checkpoint_{self.start_epoch-1}.tar"
        checkpoint_path = os.path.join(drive_path, checkpoint_file)
        if not os.path.exists(checkpoint_path):
            print(f"[!] No checkpoint for epoch {self.start_epoch -1}")
            return

        checkpoint = torch.load(checkpoint_path)

        self.generator.load_state_dict(
            checkpoint[f"generator_dict_{self.start_epoch-1}"]
        )
        print("Generator weights loaded.")

        if self.load_previous_opt:
            self.discriminator.load_state_dict(
                checkpoint[f"discriminator_dict_{self.start_epoch-1}"]
            )
            print("Discriminator weights loaded.")

            self.optimizer_generator.load_state_dict(
                checkpoint[f"optim_gen_{self.start_epoch-1}"]
            )
            self.optimizer_discriminator.load_state_dict(
                checkpoint[f"optim_dis_{self.start_epoch-1}"]
            )
            print("Optimizer's state loaded")
            try:
                self.scaler_gen = checkpoint[f"grad_scaler_gen_{self.start_epoch-1}"]
                print("Grad Scaler - Generator loaded")
                self.scaler_dis = checkpoint[f"grad_scaler_dis_{self.start_epoch-1}"]
                print("Grad Scaler - Discriminator loaded")
            except Exception as e:
                print(e)

        self.metrics["dis_loss"] = checkpoint[f"metrics_till_{self.start_epoch-1}"][
            "dis_loss"
        ]
        self.metrics["gen_loss"] = checkpoint[f"metrics_till_{self.start_epoch-1}"][
            "gen_loss"
        ]
        self.metrics["per_loss"] = checkpoint[f"metrics_till_{self.start_epoch-1}"][
            "per_loss"
        ]
        self.metrics["con_loss"] = checkpoint[f"metrics_till_{self.start_epoch-1}"][
            "con_loss"
        ]
        self.metrics["adv_loss"] = checkpoint[f"metrics_till_{self.start_epoch-1}"][
            "adv_loss"
        ]
        self.metrics["PSNR"] = checkpoint[f"metrics_till_{self.start_epoch-1}"]["PSNR"]
        self.metrics["SSIM"] = checkpoint[f"metrics_till_{self.start_epoch-1}"]["SSIM"]
        self.start_epoch = checkpoint["next_epoch"]

        print(f'Mini_batch completed: {checkpoint["steps_completed"]}')
        print(f"Checkpoint: {self.start_epoch-1} loaded")


# function ClickConnect(){
# console.log("Working");
# document.querySelector("colab-toolbar-button").click()
# }setInterval(ClickConnect,300000)

# function KeepClicking(){
# console.log("Clicking");
# document.querySelector("colab-connect-button").click()
# }
# setInterval(KeepClicking,300000)
