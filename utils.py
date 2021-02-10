from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d


def denormalize(tensors):
    """Normalization parameters for pre-trained PyTorch models
     Denormalizes image tensors using mean and std """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    tensors = tensors.clone()
    # print(tensors[:, 0, :, :].size())
    # print(tensors[:, 1, :, :].size())
    # print(tensors[:, 2, :, :].size())

    for c in range(3):
        tensors[:, c, :, :].mul_(std[c]).add_(mean[c])

    return torch.from_numpy(np.clip(tensors.cpu().numpy(), 0, 255))


def _psnr(ground, gen):
    score = psnr(ground, gen, data_range=ground.max() - ground.min())
    return round(score, 3)


def _ssim(ground, gen):
    score = ssim(gen, ground, data_range=ground.max() - ground.min(), multichannel=True)
    return round(score, 3)


def cal_img_metrics(generated, ground_truth):

    generated = generated.clone().detach().cpu()
    ground_truth = ground_truth.clone().detach().cpu()

    scores_PSNR = []
    scores_SSIM = []
    generated = denormalize(generated).permute(0, 2, 3, 1).numpy() * 255.0
    ground_truth = denormalize(ground_truth).permute(0, 2, 3, 1).numpy() * 255.0

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


# Taken from https://www.kaggle.com/ibtesama/gan-in-pytorch-with-fid


def calculate_activation_statistics(images, model, dims=2048, cuda=False):
    model.eval()
    act = np.empty((len(images), dims))

    if cuda:
        if images.device != "cuda":
            images = images.to("cuda")
    else:
        images = images

    pred = model(images)[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    act = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)

    images = images.cpu()
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def cal_fretchet(images_real, images_fake, fid_model, dims=None):
    gpu_ = torch.cuda.is_available()

    mu_1, std_1 = calculate_activation_statistics(
        images_real, fid_model, dims=dims, cuda=gpu_
    )
    mu_2, std_2 = calculate_activation_statistics(
        images_fake, fid_model, dims=dims, cuda=gpu_
    )

    """get fretched distance"""
    fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
    return fid_value
