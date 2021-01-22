from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import numpy as np


def _psnr(ground, gen):
    score = psnr(ground, gen)
    return round(score, 3)


def _ssim(ground, gen):
    score = ssim(gen, ground,
                 data_range=ground.max() - ground.min(),
                 multichannel=True)
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

    for i in range(ground_truth.size(0)):
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


if __name__ == '__main__':

    image_path_lr = r"images/lr/00000.png"
    image_path_hr = r"images/hr/00000.png"

    image_1 = cv2.imread(image_path_lr, cv2.IMREAD_COLOR)
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)

    image_2 = cv2.imread(image_path_hr, cv2.IMREAD_COLOR)
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
    # print(help(cv2.resize))
    image_2 = cv2.resize(image_2, (128, 128),
                         interpolation=cv2.INTER_CUBIC)

    image_1 = image_1.astype(np.float32)
    image_1 = image_1 // 255.0

    image_2 = image_2.astype(np.float32)
    image_2 = image_2 // 255.0

    print(cal_img_metrics(image_1, image_2))
