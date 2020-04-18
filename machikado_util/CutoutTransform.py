import numpy as np
import cv2

import shapely.geometry as geometry
from fvcore.transforms.transform import Transform
from detectron2.data import transforms as T


class CutoutTransform(Transform):
    def __init__(self, h, w, centers, radii, colors):
        """
        centers: 中心のリスト
        radii: 半径のリスト
        colors: 色のリスト
        """
        super().__init__()
        self._set_attributes(locals())
        
    def apply_image(self, img: np.ndarray):
        assert len(img.shape) == 3, '3ch のカラー画像のみを対象とする'
        h, w = img.shape[:2]
        assert (self.h == h and self.w == w), '画像サイズ不整合 h:w {}:{} -> {}:{}'.format(self.h, self.w, h, w)
        
        assert len(self.centers) == len(self.radii) == len(self.colors), '引数の不整合'
        for pt, r, c in zip(self.centers, self.radii, self.colors):
            img = cv2.circle(img, pt, r, color=c, thickness=-1)
        
        return img
        
    def apply_coords(self, coords: np.ndarray):
        return coords


class RandomCutout(T.TransformGen):
    def __init__(self, num_hole_range, radius_range, color_ranges):
        """
        num_max_hole: 穴数の範囲(min, max)
        radius_range: 半径の範囲(min, max) ※画像の短辺の長さに対する割合で指定
        color_ranges: RGBそれぞれの範囲組み合わせ [Bの範囲(min, max), Gの範囲(min, max), Rの範囲(min, max)]
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        
        short_len = h if h < w else w
        
        num_hole = np.random.randint(self.num_hole_range[0], self.num_hole_range[1])
        centers, radii, colors = [], [], []
        
        for _ in range(num_hole):
            centers += [(int(np.random.uniform(0, h)), int(np.random.uniform(0, w)))]
            radii += [int(short_len * np.random.uniform(self.radius_range[0], self.radius_range[1]))]

            b = np.random.uniform(self.color_ranges[0][0], self.color_ranges[0][1])
            g = np.random.uniform(self.color_ranges[1][0], self.color_ranges[1][1])
            r = np.random.uniform(self.color_ranges[2][0], self.color_ranges[2][1])
            colors += [(int(b), int(g), int(r))]
       
        return CutoutTransform(h, w, centers, radii, colors)