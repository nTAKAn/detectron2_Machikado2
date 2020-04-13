from detectron2.config import CfgNode as CN


def get_custom_cfg():
    """
    カムタムデータセットマッパー用の設定
    """
    _C = CN()
    _C.INPUT = CN()
    _C.INPUT.CONTRAST = CN()
    _C.INPUT.BRIGHTNESS = CN()
    _C.INPUT.SATURATION = CN()
    _C.INPUT.EXTENT = CN()
    _C.INPUT.ROTATE = CN()
    _C.INPUT.SHEAR = CN()

    # コントラストの変更
    _C.INPUT.CONTRAST.ENABLED = True
    _C.INPUT.CONTRAST.RANGE = (0.5, 1.5)
    # 明るさの変更
    _C.INPUT.BRIGHTNESS.ENABLED = True
    _C.INPUT.BRIGHTNESS.RANGE = (0.8, 1.2)
    # 彩度の変更
    _C.INPUT.SATURATION.ENABLED = True
    _C.INPUT.SATURATION.RANGE = (0.8, 1.2)
    # アフィン: 位置の変更
    _C.INPUT.EXTENT.ENABLED = True
    _C.INPUT.EXTENT.SHIFT_RANGE = (0.2, 0.2)
    # アフィン: 回転変形
    _C.INPUT.ROTATE.ENABLED = True
    _C.INPUT.ROTATE.ANGLE = [-20, 20]
    # アフィン: せん断変形
    _C.INPUT.SHEAR.ENABLED = True
    _C.INPUT.SHEAR.ANGLE_H_RANGE = (-10, 10)
    _C.INPUT.SHEAR.ANGLE_V_RANGE = (-10, 10)
    
    return _C.clone()