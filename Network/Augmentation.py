# import imageio, os
#######---------------------------------------------------------IMGAUG Data Augmentation-----------------------------------------------########
#### (https://github.com/aleju/imgaug)
#### (https://imgaug.readthedocs.io/en/latest/source/examples_basics.html)
#### (https://imgaug.readthedocs.io/en/latest/source/augmenters.html)

##### Imgs ONLY: Define a Sequence of Augmentations
##### (Operate ONLY on img pixels)
from imgaug import augmenters as iaa
import numpy as np
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class Augmentor:
    Noise = 1
    Geometric = 2  # not used anymore
    Both = 3  # not used anymore

    def __init__(self, config=0, rep=1, unlabel_included=False):
        self.apply_noise = True if (config & Augmentor.Noise) else False
        self.rep = rep
        self.unlabel_included = unlabel_included

        self.aug_ops = iaa.OneOf(
            [
                ### 1. Add gaussian noise (aka white noise) to images.
                iaa.AdditiveGaussianNoise(scale=0.01 * 255),
                ### 2. Add random values between -10 and 10 to images
                iaa.AddElementwise((-10, 10)),
                ### 3. Multiply all pixels in an image with a specific value, thereby making the image darker or brighter
                iaa.Multiply((0.8, 1.2)),
                ### 4. Sharpen an image, then overlay the results with the original using an alpha between 0.0 and 1.0
                iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.8, 1.2)),
                ### 5. Augmenter that embosses images and overlays the result with the original image
                iaa.Emboss(alpha=(0.0, 1.0), strength=(0.8, 1.2)),
                ### 6. changes the contrast of images
                iaa.ContrastNormalization((0.8, 1.2)),
                ### 7. blur images using gaussian/average/median kernels
                iaa.GaussianBlur(sigma=(0.5, 1.5)),
                ### 8. flip image vertically and/or horizontally
                # iaa.Fliplr(0.5),  # horizontal flips
                # iaa.Flipud(0.5),  # vertical flip
                ### 9. apply affine transformation
                # iaa.Affine(
                #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                #     rotate=(-180, 180),
                #     shear=(-8, 8)
                # )
            ])

        # self.aug_ops = iaa.OneOf(
        #     [
        #         ### 1. Add gaussian noise (aka white noise) to images.
        #         # iaa.AdditiveGaussianNoise(scale=0.01 * 255),
        #         ### 2. Add random values between -40 and 40 to images
        #         # iaa.AddElementwise((-10, 10)),
        #         ### 3. Multiply all pixels in an image with a specific value, thereby making the image darker or brighter
        #         iaa.Multiply((0.99, 1.01)),
        #         ### 4. Sharpen an image, then overlay the results with the original using an alpha between 0.0 and 1.0
        #         # iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.8, 1.2)),
        #         ### 5. Augmenter that embosses images and overlays the result with the original image
        #         # iaa.Emboss(alpha=(0.0, 1.0), strength=(0.8, 1.2)),
        #         ### 6. changes the contrast of images
        #         # iaa.ContrastNormalization((0.8, 1.2)),
        #         ### 7. blur images using gaussian/average/median kernels
        #         # iaa.GaussianBlur(sigma=(0.5, 1.5)),
        #         ### 8. flip image vertically and/or horizontally
        #         # iaa.Fliplr(0.5),  # horizontal flips
        #         # iaa.Flipud(0.5),  # vertical flip
        #         ### 9. apply affine transformation
        #         # iaa.Affine(
        #         #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        #         #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        #         #     rotate=(-180, 180),
        #         #     shear=(-8, 8)
        #         # )
        #     ])

    def augment(self, images, masks):
        images_aug = []
        gt_aug = []
        for img, gt in zip(images,masks):
            segmap = SegmentationMapsOnImage(gt, shape=gt.shape)
            images_aug_i, segmaps_aug_i = self.aug_ops(image=img, segmentation_maps=segmap)
            images_aug.append(images_aug_i)
            gt_aug.append(segmaps_aug_i.arr[...,0].astype(np.uint8))
        return np.stack(images_aug), np.stack(gt_aug)

    def augment_single(self, images, masks):
        if self.apply_noise is True:
            images = aug_noise.augment_images(images)
        return images, masks

    def augment_multiple(self, images, masks):
        ret_tuple = tuple()
        y_batch = None  # we don't want to repeat on ground_truth
        y_valid = True
        for rep in range(self.rep):  # this only apply to X.
            if self.apply_noise is True:
                x_batch = aug_noise.augment_images(images)
            else:
                # reduce one copy op if it's last iteration
                x_batch = images if (rep + 1 == self.rep) else images.copy()
            if y_batch is None:
                y_batch = masks
            ret_tuple += (x_batch,)

        if y_batch.shape[0] == 0:  # it can be 0 for unlabeled data.
            # this is necessary because pytorch can't accept NoneType, empty (return without y_batch) or zero rows (0, 256,26,1) etc...
            y_batch = np.empty((1,) + y_batch.shape[1:], dtype=np.float32)
            y_valid = False
        return ret_tuple, (y_batch, y_valid)