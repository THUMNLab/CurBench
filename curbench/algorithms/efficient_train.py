from copy import deepcopy

import torch
import torch.nn.functional as F
from torchvision import transforms
from timm.data import create_transform, auto_augment

from .base import BaseTrainer, BaseCL


class EfficientTrain(BaseCL):
    """EfficientTrain

    EfficientTrain: Exploring Generalized Curriculum Learning for Training Visual Backbones
    https://arxiv.org/pdf/2211.09703.pdf
    """

    class FreqCrop(object):
        def __init__(self, band_width):
            super().__init__()
            self.band_width = band_width

        def _freq_crop(self, input_t):
            img_size = input_t.size(-1)
            band_w = self.band_width // 2

            img_f = torch.fft.fft2(input_t)

            img_crop = torch.empty(
                [input_t.size(0), band_w * 2, band_w * 2],
                dtype=img_f.dtype, device=img_f.device
            )

            img_crop[:, :band_w, :band_w] = img_f[:, :band_w, :band_w]
            img_crop[:, -band_w:, :band_w] = img_f[:, -band_w:, :band_w]
            img_crop[:, :band_w, -band_w:] = img_f[:, :band_w, -band_w:]
            img_crop[:, -band_w:, -band_w:] = img_f[:, -band_w:, -band_w:]

            img_crop = img_crop * ((band_w * 2 / img_size) ** 2)

            img_crop = torch.fft.ifft2(img_crop)
            img_crop = torch.real(img_crop)

            img_crop = F.interpolate(img_crop.unsqueeze(0), size=img_size, mode='bicubic', align_corners=False).squeeze(0)

            return img_crop

        def __call__(self, img):
            return self._freq_crop(img)


    def __init__(self, grow_epochs):
        super(EfficientTrain, self).__init__()

        self.name = 'efficient_train'
        self.epoch = 0

        self.grow_epochs = grow_epochs


    def data_prepare(self, loader, **kwargs):
        super().data_prepare(loader)

        self.raw_dataset = self._get_raw_dataset(self.dataset)
        self.transform = deepcopy(self.raw_dataset.transform)
        self.image_size = self.raw_dataset.image_size

        self.e_list = [60, 120, 180, 240, 300]
        self.e_list = [int(e / 300 * self.grow_epochs) for e in self.e_list]

        self.b_list = [160, 160, 160, 192, 224]
        self.b_list = [int(b / 224 * self.image_size) for b in self.b_list]


    def data_curriculum(self, **kwargs):
        self.epoch += 1
        
        if (self.epoch - 1) % self.e_list[0] == 0:
            ET_index = (self.epoch - 1) // self.e_list[0]

            # weaker-tostronger randaug
            aa = 'rand-m%d-mstd0.5-inc1' % (2 * ET_index + 1)
            timm_transform = create_transform(
                input_size=self.image_size,
                is_training=True,
                auto_augment=aa,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            auto_augment_transform = [t for t in timm_transform.transforms if isinstance(t, auto_augment.AutoAugment)]
            
            # low-frequency cropping
            if self.b_list[ET_index] < self.image_size:
                freq_crop_transform = [self.FreqCrop(self.b_list[ET_index])]
            else:
                freq_crop_transform = []

            self.raw_dataset.transform = transforms.Compose([
                *auto_augment_transform,
                *self.transform.transforms,
                *freq_crop_transform,
            ])
        return self._dataloader(self.dataset)


    def _get_raw_dataset(self, dataset):
        if hasattr(dataset, 'dataset'):
            return self._get_raw_dataset(dataset.dataset)
        return dataset


class EfficientTrainTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, gpu_index, num_epochs, random_seed,
                 grow_epochs):
        
        cl = EfficientTrain(grow_epochs)
        
        super(EfficientTrainTrainer, self).__init__(
            data_name, net_name, gpu_index, num_epochs, random_seed, cl)