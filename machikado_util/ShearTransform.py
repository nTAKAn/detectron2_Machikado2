import numpy as np
import cv2

import shapely.geometry as geometry
from fvcore.transforms.transform import Transform
from detectron2.data import transforms as T


class ShearTransform(Transform):
    """
    せん断変形を行うカスタムトランスフォーム
    """
    def __init__(self, h: int, w: int, angle_h: float, angle_v: float):
        super().__init__()
        self._set_attributes(locals())
        
        self.mat = np.array([[1, np.tan(np.deg2rad(angle_h)), 0],
                             [np.tan(np.deg2rad(angle_v)), 1, 0]])
        
    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """イメージの変形を行う
        
        Arguments:
            img {np.ndarray} -- 元イメージ
        
        Returns:
            np.ndarray -- 変形されたイメージ
        """
        assert len(img.shape) == 3, '3ch のカラー画像のみを対象とする'
        
        h, w = img.shape[:2]
        assert (self.h == h and self.w == w), '画像サイズ不整合 h:w {}:{} -> {}:{}'.format(self.h, self.w, h, w)
        
        return cv2.warpAffine(img, self.mat, (w, h))

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """領域座標変換
        
        Arguments:
            coords {np.ndarray} -- 変換した座標
        
        Returns:
            np.ndarray -- 変換された座標
        """
        p = np.vstack([coords.T, np.ones((1, len(coords)))])
        p = np.dot(self.mat, p)
        
        return p.T

    def apply_polygons(self, polygons: list) -> list:
        """ポリゴン（領域データの変換）
        
        Arguments:
            polygons {list} -- [description]
        
        Returns:
            list -- [description]
        """
        polygons = [self.apply_coords(p) for p in polygons]

        # 画像範囲でクリッピング
        crop_box = geometry.box(0, 0, self.w, self.h).buffer(0)
        
        cropped_polygons = []
        
        for polygon in polygons:
            polygon = geometry.Polygon(polygon).buffer(0)
            assert polygon.is_valid, '不正なポリゴン {}'.format(polygon)
            
            cropped = polygon.intersection(crop_box)
            
            if cropped.is_empty:
                continue
            
            # 複数のポリゴンに分割される可能性があるのでその処理が必要
            if not isinstance(cropped, geometry.collection.BaseMultipartGeometry):  # 単一であればリストに
                cropped = [cropped]
            
            for poly in cropped:
                if not isinstance(poly, geometry.Polygon) or not poly.is_valid:  # 不正なポリンゴンを無視する
                    continue
                
                coords = np.asarray(poly.exterior.coords)
                cropped_polygons.append(coords[:-1])  # ポリゴンの終端が先端になっているので終端を削除
        
        if len(cropped_polygons) == 0:
            print('警告: せん断変形の結果、有効な領域が残らなかった')

#        assert len(cropped_polygons) > 0, 'せん断変形の結果、有効な領域が残らなかった'
        
        return cropped_polygons


class RandomShear(T.TransformGen):
    """
    ランダムにせん断変形するジェネレータ
    """
    def __init__(self, angle_h_range, angle_v_range):
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        
        if self.angle_h_range is None:
            angle_h = 0
        else:
            angle_h = np.random.uniform(self.angle_h_range[0], self.angle_h_range[1])
            
        if self.angle_v_range is None:
            angle_v = 0
        else:
            angle_v = np.random.uniform(self.angle_v_range[0], self.angle_v_range[1])
       
        return ShearTransform(h, w, angle_h, angle_v)