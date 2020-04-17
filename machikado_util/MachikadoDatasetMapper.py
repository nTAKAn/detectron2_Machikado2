import copy
import logging
import numpy as np
import torch

from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

from .ShearTransform import ShearTransform, RandomShear
from .CutoutTransform import CutoutTransform, RandomCutout

class MachikadoDatasetMapper:
    """
    カスタムデータマッパー
    """
    def __init__(self, cfg, is_train=True):
        assert cfg.MODEL.MASK_ON, '今回はセグメンテーションのみを対象にする'
        assert not cfg.MODEL.KEYPOINT_ON, 'キーポイントは扱わない'
        assert not cfg.MODEL.LOAD_PROPOSALS, 'pre-computed proposals っていうのがよくわからん・・・・とりあえず無効前提で'
        
        self.cont_gen = None
        if cfg.INPUT.CONTRAST.ENABLED:
            self.cont_gen = T.RandomContrast(cfg.INPUT.CONTRAST.RANGE[0], cfg.INPUT.CONTRAST.RANGE[1])
            logging.getLogger(__name__).info('ContGen used in training.')
            
        self.bright_gen = None
        if cfg.INPUT.BRIGHTNESS.ENABLED:
            self.bright_gen = T.RandomBrightness(cfg.INPUT.BRIGHTNESS.RANGE[0], cfg.INPUT.BRIGHTNESS.RANGE[1])
            logging.getLogger(__name__).info('BrightGen used in training.')

        self.sat_gen = None
        if cfg.INPUT.SATURATION.ENABLED:
            self.sat_gen = T.RandomSaturation(cfg.INPUT.SATURATION.RANGE[0], cfg.INPUT.SATURATION.RANGE[1])
            logging.getLogger(__name__).info('SatGen used in training.')
        
        self.cutout_gen = None
        if cfg.INPUT.CUTOUT.ENABLED:
            self.cutout_gen = RandomCutout(cfg.INPUT.CUTOUT.NUM_HOLE_RANGE, cfg.INPUT.CUTOUT.RADIUS_RANGE, cfg.INPUT.CUTOUT.COLOR_RANGE)
            logging.getLogger(__name__).info('CutoutGen used in training.')

        self.extent_gen = None
        if cfg.INPUT.EXTENT.ENABLED:
            self.extent_gen = T.RandomExtent(scale_range=(1, 1), shift_range=cfg.INPUT.EXTENT.SHIFT_RANGE)
            logging.getLogger(__name__).info('ExtentGen used in training.')
        
        self.crop_gen = None
        if cfg.INPUT.CROP.ENABLED:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info('CropGen used in training: ' + str(self.crop_gen))
        
        self.rotate_gen = None
        if cfg.INPUT.ROTATE.ENABLED:
            self.rotate_gen = T.RandomRotation(cfg.INPUT.ROTATE.ANGLE, expand=False)
            logging.getLogger(__name__).info('RotateGen used in training.')
        
        self.shear_gen = None
        if cfg.INPUT.SHEAR.ENABLED:
            self.shear_gen = RandomShear(cfg.INPUT.SHEAR.ANGLE_H_RANGE, cfg.INPUT.SHEAR.ANGLE_V_RANGE)
            logging.getLogger(__name__).info('ShearGen used in training.')

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)
        
        self.img_format = cfg.INPUT.FORMAT
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.is_train = is_train

    def __call__(self, dataset_dict):
        assert 'annotations' in dataset_dict, '今回はセグメンテーションのみを対象にする'
        assert not 'sem_seg_file_name' in dataset_dict, 'パノプティックセグメンテーションは行わない'
        
        dataset_dict = copy.deepcopy(dataset_dict)
        
        image = utils.read_image(dataset_dict['file_name'], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        
        # テストの場合はアノテーションがいらないので削除して終了
        if not self.is_train:
            dataset_dict.pop('annotations', None)
            dataset_dict.pop('sem_seg_file_name', None)
            return dataset_dict
        
        # 明るさ・コントラスト・彩度・カットアウト
        if self.cont_gen is not None:
            tfm = self.cont_gen.get_transform(image)
            image = tfm.apply_image(image)
        if self.bright_gen is not None:
            tfm = self.bright_gen.get_transform(image)
            image = tfm.apply_image(image)
        if self.sat_gen is not None:
            tfm = self.sat_gen.get_transform(image)
            image = tfm.apply_image(image)
        if self.cutout_gen is not None:
            tfm = self.cutout_gen.get_transform(image)
            image = tfm.apply_image(image)
            
        # アフィン
        if self.rotate_gen is not None:
            rotate_tfm = self.rotate_gen.get_transform(image)
            image = rotate_tfm.apply_image(image)
        if self.shear_gen is not None:
            shear_tfm = self.shear_gen.get_transform(image)
            image = shear_tfm.apply_image(image)
        if self.extent_gen is not None:
            extent_tfm = self.extent_gen.get_transform(image)
            image = extent_tfm.apply_image(image)
        if self.crop_gen is not None:
            crop_tfm = utils.gen_crop_transform_with_instance(
                self.crop_gen.get_crop_size(image.shape[:2]), image.shape[:2], np.random.choice(dataset_dict['annotations']))
            image = crop_tfm.apply_image(image)
        
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        
        if self.crop_gen is not None:
            transforms = crop_tfm + transforms
        if self.extent_gen is not None:
            transforms = extent_tfm + transforms
        if self.shear_gen is not None:
            transforms = shear_tfm + transforms
        if self.rotate_gen is not None:
            transforms = rotate_tfm + transforms

        image_shape = image.shape[:2]  # h, w
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        annos = [utils.transform_instance_annotations(obj, transforms, image_shape, keypoint_hflip_indices=None)
                 for obj in dataset_dict.pop('annotations')
                 if obj.get("iscrowd", 0) == 0]

        instances = utils.annotations_to_instances(annos, image_shape, mask_format=self.mask_format)

        # マスクからバウンディングボックスを作成
        if self.crop_gen and instances.has("gt_masks"):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict