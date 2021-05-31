import os
import shutil
import numpy as np
import gc
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.cuda import amp
from torch.optim.adam import Adam
from torch.autograd import Variable

from torchvision.utils import save_image

from models import PerceptualLoss, Generator, Discriminator, InceptionV3
from utils import cal_img_metrics, denormalize, cal_fretchet

import wandb


class Trainer:
    def __init__(self, config, data_loader_train, data_loader_val, device):
        self.device = device
        self.num_epoch = config["num_epoch"]
        self.start_epoch = config["start_epoch"]
        self.image_size = config["image_size"]
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

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        # FID score
        FID_DIM = 768
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[FID_DIM]
        fid_model = InceptionV3([block_idx])
        fid_model.to(self.device)

        best_val_psnr = 0.0
        best_val_ssim = 0.0
        best_val_fid = np.inf

        # FOR BICUBIC
        already_calculated_bicubic = False
        upsampler = torch.nn.Upsample(scale_factor=4, mode="bicubic")

        for epoch in range(self.start_epoch, self.start_epoch + self.num_epoch):
            self.generator.train()
            self.discriminator.train()

            epoch_gen_loss = []
            epoch_dis_loss = []
            epoch_per_loss = []
            epoch_adv_loss = []
            epoch_con_loss = []

            SAVE = False

            training_loader_iter = iter(self.data_loader_train)
            length_train = len(training_loader_iter)

            if not os.path.exists(os.path.join(self.sample_dir, str(epoch))):
                os.makedirs(os.path.join(self.sample_dir, str(epoch)))

            for step in tqdm(
                range(length_train),
                desc=f"Epoch: {epoch}/{self.start_epoch + self.num_epoch-1}",
            ):
                image = next(training_loader_iter)
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
                        for p in self.discriminator.parameters():
                            p.requires_grad = False

                        # Extract validity predictions from discriminator
                        score_real = self.discriminator(high_resolution).detach()
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
                            perceptual_loss * self.perceptual_loss_factor
                            + adversarial_loss * self.adversarial_loss_factor
                            + content_loss * self.content_loss_factor
                        )

                    else:
                        generator_loss = content_loss * self.content_loss_factor

                self.scaler_gen.scale(generator_loss).backward()
                self.scaler_gen.step(self.optimizer_generator)

                scale_gen = self.scaler_gen.get_scale()
                self.scaler_gen.update()
                skip_gen_lr_sched = scale_gen != self.scaler_gen.get_scale()
                # self.optimizer_generator.step()

                self.metrics["gen_loss"].append(
                    np.round(generator_loss.detach().item(), 5)
                )
                self.metrics["con_loss"].append(
                    np.round(content_loss.detach().item() * self.content_loss_factor, 4)
                )

                epoch_gen_loss.append(self.metrics["gen_loss"][-1])
                epoch_con_loss.append(self.metrics["con_loss"][-1])

                ##########################
                # training discriminator #
                ##########################
                if not self.is_psnr_oriented:
                    self.optimizer_discriminator.zero_grad()
                    for p in self.discriminator.parameters():
                        p.requires_grad = True

                    with amp.autocast():
                        score_real = self.discriminator(high_resolution)

                        score_fake = self.discriminator(fake_high_resolution).detach()

                        # real
                        discriminator_rf = score_real - score_fake.mean(
                            axis=0, keepdim=True
                        )
                        adversarial_loss_rf = (
                            adversarial_criterion(discriminator_rf, real_labels) * 0.5
                        )

                        # fake
                        score_fake = self.discriminator(fake_high_resolution.detach())

                        discriminator_fr = score_fake - score_real.mean(
                            axis=0, keepdim=True
                        )
                        adversarial_loss_fr = (
                            adversarial_criterion(discriminator_fr, fake_labels) * 0.5
                        )

                        # score_real = self.discriminator(high_resolution)
                        # score_fake = self.discriminator(fake_high_resolution.detach())
                        # discriminator_rf = score_real - score_fake.mean(
                        #     axis=0, keepdim=True
                        # )

                        # adversarial_loss_rf = adversarial_criterion(
                        #     discriminator_rf, real_labels
                        # )

                        # discriminator_loss = (
                        #     adversarial_loss_fr + adversarial_loss_rf
                        # ) / 2

                    self.scaler_dis.scale(adversarial_loss_rf).backward(
                        retain_graph=True
                    )
                    self.scaler_dis.scale(adversarial_loss_fr).backward()
                    self.scaler_dis.step(self.optimizer_discriminator)
                    dis_scale_val = self.scaler_dis.get_scale()
                    self.scaler_dis.update()
                    skip_dis_lr_sched = dis_scale_val != self.scaler_dis.get_scale()

                    discriminator_loss = (
                        adversarial_loss_rf.detach().item()
                        + adversarial_loss_fr.detach().item()
                    )

                    self.metrics["dis_loss"].append(np.round(discriminator_loss, 5))

                    # generator metrics
                    self.metrics["adv_loss"].append(
                        np.round(
                            adversarial_loss.detach().item()
                            * self.adversarial_loss_factor,
                            4,
                        )
                    )
                    self.metrics["per_loss"].append(
                        np.round(
                            perceptual_loss.detach().item()
                            * self.perceptual_loss_factor,
                            4,
                        )
                    )

                    epoch_dis_loss.append(self.metrics["dis_loss"][-1])
                    epoch_adv_loss.append(self.metrics["adv_loss"][-1])
                    epoch_per_loss.append(self.metrics["per_loss"][-1])

                torch.cuda.empty_cache()
                gc.collect()

                if step == int(total_step / 2) or step == 0 or step == (total_step - 1):
                    if not self.is_psnr_oriented:
                        print(
                            f"[Epoch {epoch}/{self.start_epoch+self.num_epoch-1}] [Batch {step+1}/{total_step}]"
                            f"[D loss {self.metrics['dis_loss'][-1]}] [G loss {self.metrics['gen_loss'][-1]}]"
                            f"[perceptual loss {self.metrics['per_loss'][-1]}]"
                            f"[adversarial loss {self.metrics['adv_loss'][-1]}]"
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
                            denormalize(high_resolution.detach().cpu()),
                            denormalize(upsampler(low_resolution).detach().cpu()),
                            denormalize(fake_high_resolution.detach().cpu()),
                        ),
                        2,
                    )

                    # print(result[0][:, 512:, :].min(), result[0][:, 512:, :].max())

                    save_image(
                        result,
                        os.path.join(self.sample_dir, str(epoch), f"ESR_{step+1}.png"),
                        nrow=8,
                        normalize=False,
                    )
                    wandb.log(
                        {
                            f"training_images_ESR_{step+1}": wandb.Image(os.path.join(self.sample_dir, str(epoch), f"ESR_{step+1}.png"))
                        }
                    )  

                torch.cuda.empty_cache()
                gc.collect()

            # epoch metrics
            if not self.is_psnr_oriented:
                print(
                    f"Epoch: {epoch} -> Dis loss: {np.round(np.array(epoch_dis_loss).mean(), 4)} "
                    f"Gen loss: {np.round(np.array(epoch_gen_loss).mean(), 4)} "
                    f"Per loss:: {np.round(np.array(epoch_per_loss).mean(), 4)} "
                    f"Adv loss:: {np.round(np.array(epoch_adv_loss).mean(), 4)} "
                    f"Con loss:: {np.round(np.array(epoch_con_loss).mean(), 4)}"
                    f""
                )
                wandb.log(
                    {
                        "epoch": epoch,
                        "Dis_loss": np.round(np.array(epoch_dis_loss).mean(), 4),
                        "Gen_loss": np.round(np.array(epoch_gen_loss).mean(), 4),
                        "Con_loss": np.round(np.array(epoch_con_loss).mean(), 4),
                        "Per_loss": np.round(np.array(epoch_per_loss).mean(), 4),
                        "Adv_loss": np.round(np.array(epoch_adv_loss).mean(), 4),
                        "Con_loss": np.round(np.array(epoch_con_loss).mean(), 4),
                    }
                )
            else:
                print(
                    f"Epoch: {epoch} -> "
                    f"Gen loss: {np.round(np.array(epoch_gen_loss).mean(), 4)} "
                    f"Con loss:: {np.round(np.array(epoch_con_loss).mean(), 4)}"
                    f""
                )
                wandb.log(
                    {
                        "epoch": epoch,
                        "Gen_loss": np.round(np.array(epoch_gen_loss).mean(), 4),
                        "Con_loss": np.round(np.array(epoch_con_loss).mean(), 4),
                    }
                )
                

            if not skip_gen_lr_sched:
                self.lr_scheduler_generator.step()

            if not self.is_psnr_oriented:
                if not skip_dis_lr_sched:
                    self.lr_scheduler_discriminator.step()

            # validation set SSIM and PSNR
            val_batch_psnr = []
            val_batch_ssim = []
            val_batch_FID = []

            if not already_calculated_bicubic:
                ups_batch_psnr = []
                ups_batch_ssim = []
                ups_batch_FID = []

            for idx, image_val in enumerate(self.data_loader_val):
                val_low_resolution = image_val["lr"].to(self.device)
                val_high_resolution = image_val["hr"].to(self.device)

                self.generator.eval()

                with torch.no_grad():
                    with amp.autocast():
                        val_fake_high_res = self.generator(val_low_resolution).detach()

                        # generated image metrics FID, PSNR, SSIM
                        val_fid = cal_fretchet(
                            val_high_resolution.detach(),
                            val_fake_high_res,
                            fid_model,
                            dims=FID_DIM,
                        )

                    val_psnr, val_ssim = cal_img_metrics(
                        val_fake_high_res, val_high_resolution,
                    )

                    val_batch_psnr.append(val_psnr)
                    val_batch_ssim.append(val_ssim)
                    val_batch_FID.append(val_fid)

                    # bicubic image metrics FID, PSNR, SSIM
                    if not already_calculated_bicubic:
                        with amp.autocast():
                            ups_fid = cal_fretchet(
                                val_high_resolution,
                                val_low_resolution.detach(),
                                fid_model,
                                FID_DIM,
                            )

                        ups_psnr, ups_ssim = cal_img_metrics(
                            upsampler(val_low_resolution.cpu()),
                            val_high_resolution.cpu(),
                        )

                        ups_batch_psnr.append(ups_psnr)
                        ups_batch_ssim.append(ups_ssim)
                        ups_batch_FID.append(ups_fid)

                # visualization
                result_val = torch.cat(
                    (
                        denormalize(val_high_resolution.detach().cpu()),
                        denormalize(upsampler(val_low_resolution).detach().cpu()),
                        denormalize(val_fake_high_res.detach().cpu()),
                    ),
                    2,
                )
                save_image(
                    result_val,
                    os.path.join(self.sample_dir, f"Validation_{epoch}_{idx}.png"),
                    nrow=8,
                    normalize=False,
                )

            val_epoch_psnr = round(sum(val_batch_psnr) / len(val_batch_psnr), 4)
            val_epoch_ssim = round(sum(val_batch_ssim) / len(val_batch_ssim), 4)
            val_epoch_fid = round(sum(val_batch_FID) / len(val_batch_FID), 4)

            if not already_calculated_bicubic:
                ups_epoch_psnr = round(sum(ups_batch_psnr) / len(ups_batch_psnr), 4)
                ups_epoch_ssim = round(sum(ups_batch_ssim) / len(ups_batch_ssim), 4)
                ups_epoch_FID = round(sum(ups_batch_FID) / len(ups_batch_FID), 4)
                already_calculated_bicubic = True

            # log validation psnr, ssim
            wandb.log(
                {
                    "epoch": epoch,
                    "valid_psnr": val_epoch_psnr,
                    "valid_ssim": val_epoch_ssim,
                    "valid_fid": val_epoch_fid,
                }
            )
            # log validation image 0
            wandb.log(
                {
                    "validation_images_1": wandb.Image(
                        os.path.join(self.sample_dir, f"Validation_{epoch}_0.png")
                    ),
                 "validation_images_2": wandb.Image(
                        os.path.join(self.sample_dir, f"Validation_{epoch}_1.png")
                    ),
                 "validation_images_3": wandb.Image(
                        os.path.join(self.sample_dir, f"Validation_{epoch}_2.png")
                    )
                }
            )
            
            
            if val_epoch_fid < best_val_fid:
                best_val_fid = val_epoch_fid
                SAVE = True

            if val_epoch_psnr > best_val_psnr:
                best_val_psnr = val_epoch_psnr
                SAVE = True

            if val_epoch_ssim > best_val_ssim:
                best_val_ssim = val_epoch_ssim
                SAVE = True

            self.metrics["PSNR"].append(val_epoch_psnr)
            self.metrics["SSIM"].append(val_epoch_ssim)

            print(
                f"Validation Set: PSNR: {val_epoch_psnr}, SSIM: {val_epoch_ssim}, FID: {val_epoch_fid}"
            )
            print(
                f"Bicubic Ups: PSNR: {ups_epoch_psnr}, SSIM: {ups_epoch_ssim}, FID: {ups_epoch_FID}"
            )

            del (
                val_fid,
                val_psnr,
                val_ssim,
                val_epoch_psnr,
                val_epoch_ssim,
                val_epoch_fid,
                val_low_resolution,
                val_fake_high_res,
                val_high_resolution,
            )

            torch.cuda.empty_cache()
            gc.collect()

            models_dict = {
                "next_epoch": epoch + 1,
                f"generator_dict": self.generator.state_dict(),
                f"optim_gen": self.optimizer_generator.state_dict(),
                f"grad_scaler_gen": self.scaler_gen.state_dict(),
                f"metrics": self.metrics,
            }

            if not self.is_psnr_oriented:
                models_dict[f"discriminator_dict"] = self.discriminator.state_dict()
                models_dict[f"optim_dis"] = self.optimizer_discriminator.state_dict()
                models_dict[f"grad_scaler_dis"] = self.scaler_dis.state_dict()

            save_name = f"checkpoint_{epoch}.tar"

            if SAVE:
                # remove all previous best checkpoints
                _ = [
                    os.remove(os.path.join(r"/content/", file))
                    for file in os.listdir(r"/content/")
                    if file.startswith("best_")
                ]

                _ = [
                    os.remove(
                        os.path.join(r"/content/drive/MyDrive/Project-ESRGAN", file)
                    )
                    for file in os.listdir(r"/content/drive/MyDrive/Project-ESRGAN")
                    if file.startswith("best_")
                ]
                save_name = f"best_checkpoint_{epoch}.tar"
                print(
                    f"Best val scores  till epoch {epoch} -> PSNR: {best_val_psnr}, SSIM: {best_val_ssim}, , FID: {best_val_fid}"
                )

            if save_name.startswith("checkpoint"):
                _ = [
                    os.remove(os.path.join(r"/content/", file))
                    for file in os.listdir(r"/content/")
                    if file.startswith("checkpoint")
                ]
                _ = [
                    os.remove(
                        os.path.join(r"/content/drive/MyDrive/Project-ESRGAN", file)
                    )
                    for file in os.listdir(r"/content/drive/MyDrive/Project-ESRGAN")
                    if file.startswith("checkpoint")
                ]

            torch.save(models_dict, save_name)
            shutil.copyfile(
                save_name,
                os.path.join(r"/content/drive/MyDrive/Project-ESRGAN", save_name),
            )

            torch.cuda.empty_cache()
            gc.collect()

        wandb.run.summary["valid_psnr"] = best_val_psnr
        wandb.run.summary["valid_ssim"] = best_val_ssim
        wandb.run.summary["valid_fid"] = best_val_fid

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
            lr=0.0004,
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

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.generator.load_state_dict(checkpoint[f"generator_dict"])
        print("Generator weights loaded.")

        if self.load_previous_opt:
            self.optimizer_generator.load_state_dict(checkpoint[f"optim_gen"])
            print("Generator Optimizer state loaded")

            self.scaler_gen.load_state_dict(checkpoint[f"grad_scaler_gen"])
            print("Grad Scaler - Generator loaded")

            try:
                self.discriminator.load_state_dict(checkpoint[f"discriminator_dict"])
                print("Discriminator weights loaded.")
                self.optimizer_discriminator.load_state_dict(checkpoint[f"optim_dis"])
                print("Discriminator optimizer loaded.")

                self.scaler_dis.load_state_dict(checkpoint[f"grad_scaler_dis"])
                print("Grad Scaler - Discriminator loaded")
            except:
                pass

        self.metrics["dis_loss"] = checkpoint[f"metrics"]["dis_loss"]
        self.metrics["gen_loss"] = checkpoint[f"metrics"]["gen_loss"]
        self.metrics["per_loss"] = checkpoint[f"metrics"]["per_loss"]
        self.metrics["con_loss"] = checkpoint[f"metrics"]["con_loss"]
        self.metrics["adv_loss"] = checkpoint[f"metrics"]["adv_loss"]
        self.metrics["PSNR"] = checkpoint[f"metrics"]["PSNR"]
        self.metrics["SSIM"] = checkpoint[f"metrics"]["SSIM"]
        self.start_epoch = checkpoint["next_epoch"]

        temp = []

        if self.decay_iter:
            self.decay_iter = np.array(self.decay_iter) - self.start_epoch

            for i in self.decay_iter:
                if i > 0:
                    temp.append(i)

        if not temp:
            temp.append(200)

        self.decay_iter = temp
        print("Decay_iter:", self.decay_iter)

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
