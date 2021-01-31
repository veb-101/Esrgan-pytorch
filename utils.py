# Normalization parameters for pre-trained PyTorch models
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import torch


def denormalize(tensors):
    """Normalization parameters for pre-trained PyTorch models
     Denormalizes image tensors using mean and std """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    tensors = tensors.clone().detach()
    # print(tensors[:, 0, :, :].size())
    # print(tensors[:, 1, :, :].size())
    # print(tensors[:, 2, :, :].size())

    for c in range(3):
        tensors[:, c, :, :].mul_(std[c]).add_(mean[c])
        # print(tensors.min(), tensors.max())
    # return torch.clamp(np.clip(tensors.numpy(), 0, 255))
    return torch.from_numpy(np.clip(tensors.cpu().detach().numpy(), 0, 255))


def _psnr(ground, gen):
    score = psnr(ground, gen, data_range=ground.max() - ground.min())
    return round(score, 3)


def _ssim(ground, gen):
    score = ssim(gen, ground, data_range=ground.max() - ground.min(), multichannel=True)
    return round(score, 3)


def cal_img_metrics(generated, ground_truth):

    generated = generated.clone().detach()
    ground_truth = ground_truth.clone().detach()

    scores_PSNR = []
    scores_SSIM = []
    generated = denormalize(generated).permute(0, 2, 3, 1).numpy()
    ground_truth = denormalize(ground_truth).permute(0, 2, 3, 1).numpy()

    # gen = gen.permute(0, 2, 3, 1).numpy() * 255.0
    # ground = ground.permute(0, 2, 3, 1).numpy() * 255.0

    for i in range(len(ground_truth)):
        ground = ground_truth[i]
        gen = generated[i]

        # print(ground_truth.max() - ground_truth.min())
        psnr_ = _psnr(ground, gen)
        ssim_ = _ssim(ground, gen)

        scores_PSNR.append(psnr_)
        scores_SSIM.append(ssim_)

    return (
        round(sum(scores_PSNR) / len(scores_PSNR), 3),
        round(sum(scores_SSIM) / len(scores_SSIM), 3),
    )


# def denormalize(tensors):
#     """Normalization parameters for pre-trained PyTorch models
#      Denormalizes image tensors using mean and std """
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])

#     tensors = tensors.clone().cpu().detach()
#     tensors = tensors.permute(0, 2, 3, 1).numpy()
#     # row, height, width, channel

#     with torch.no_grad():
#         for image in tensors:
#             for c in range(3):
#                 np.add(np.multiply(image[:, :, c], std[c]), mean[c])

#     # row, channel, height, width
#     tensors = np.moveaxis(tensors, (0, 3, 1, 2), (0, 1, 2, 3))
#     # print(tensors.shape)
#     return torch.from_numpy(np.clip(tensors, 0, 255))

# def ssim(img1, img2):
#     C1 = (0.01 * 255) ** 2
#     C2 = (0.03 * 255) ** 2
#
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())
#
#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1 ** 2
#     mu2_sq = mu2 ** 2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
#
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
#         (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
#     )
#     return ssim_map.mean()
#
#
# def calculate_ssim(img1, img2):
#     """calculate SSIM
#     the same outputs as MATLAB's
#     img1, img2: [0, 255]
#     """
#     if not img1.shape == img2.shape:
#         raise ValueError("Input images must have the same dimensions.")
#     if img1.ndim == 2:
#         return ssim(img1, img2)
#     elif img1.ndim == 3:
#         if img1.shape[2] == 3:
#             ssims = []
#             for i in range(3):
#                 ssims.append(ssim(img1, img2))
#             return np.array(ssims).mean()
#         elif img1.shape[2] == 1:
#             return ssim(np.squeeze(img1), np.squeeze(img2))
#     else:
#         raise ValueError("Wrong input image dimensions.")
#
#
# def calculate_psnr(img1, img2):
#     # img1 and img2 have range [0, 255]
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return float("inf")
#     return 20 * np.log10(255.0 / np.sqrt(mse))
#
#
# def cal_img_metrics(gen, ground):
#     with torch.no_grad():
#         scores_PSNR = []
#         scores_SSIM = []
#         gen = denormalize(gen).permute(0, 2, 3, 1).numpy() * 255.0
#         ground = denormalize(ground).permute(0, 2, 3, 1).numpy() * 255.0
#
#         # gen = gen.permute(0, 2, 3, 1).numpy() * 255.0
#         # ground = ground.permute(0, 2, 3, 1).numpy() * 255.0
#
#         for generated, ground_truth in zip(gen, ground):
#             # print(ground_truth.max() - ground_truth.min())
#             psnr_ = calculate_psnr(generated, ground_truth)
#             ssim_ = calculate_ssim(generated, ground_truth)
#
#             scores_PSNR.append(psnr_)
#             scores_SSIM.append(ssim_)
#
#         return (
#             round(sum(scores_PSNR) / len(scores_PSNR), 3),
#             round(sum(scores_SSIM) / len(scores_SSIM), 3),
#         )


# if __name__ == '__main__':

#     image_path_lr = r"images/lr/00000.png"
#     image_path_hr = r"images/hr/00000.png"

#     image_1 = cv2.imread(image_path_lr, cv2.IMREAD_COLOR)
#     image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)

#     image_2 = cv2.imread(image_path_hr, cv2.IMREAD_COLOR)
#     image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
#     # print(help(cv2.resize))
#     image_2 = cv2.resize(image_2, (128, 128),
#                          interpolation=cv2.INTER_CUBIC)

#     image_1 = image_1.astype(np.float32)
#     image_1 = image_1 // 255.0

#     image_2 = image_2.astype(np.float32)
#     image_2 = image_2 // 255.0

#     print(cal_img_metrics(image_1, image_2))
