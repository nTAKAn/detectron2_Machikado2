from detectron2.config import CfgNode as CN


def append_custom_cfg(cfg):
    """
    カムタムデータセットマッパー用の設定を追加する
    """
    cfg.INPUT.CONTRAST = CN()
    cfg.INPUT.BRIGHTNESS = CN()
    cfg.INPUT.SATURATION = CN()
    cfg.INPUT.EXTENT = CN()
    cfg.INPUT.ROTATE = CN()
    cfg.INPUT.SHEAR = CN()
    cfg.INPUT.CUTOUT = CN()

    # コントラストの変更
    cfg.INPUT.CONTRAST.ENABLED = True
    cfg.INPUT.CONTRAST.RANGE = (0.5, 1.5)
    # 明るさの変更
    cfg.INPUT.BRIGHTNESS.ENABLED = True
    cfg.INPUT.BRIGHTNESS.RANGE = (0.8, 1.2)
    # 彩度の変更
    cfg.INPUT.SATURATION.ENABLED = True
    cfg.INPUT.SATURATION.RANGE = (0.8, 1.2)
    # アフィン: 位置の変更
    cfg.INPUT.EXTENT.ENABLED = True
    cfg.INPUT.EXTENT.SHIFT_RANGE = (0.2, 0.2)
    # アフィン: 回転変形
    cfg.INPUT.ROTATE.ENABLED = True
    cfg.INPUT.ROTATE.ANGLE = [-10, 10]
    # アフィン: せん断変形
    cfg.INPUT.SHEAR.ENABLED = True
    cfg.INPUT.SHEAR.ANGLE_H_RANGE = (-10, 10)
    cfg.INPUT.SHEAR.ANGLE_V_RANGE = (-10, 10)
    # カットアウト
    cfg.INPUT.CUTOUT.ENABLED = True
    cfg.INPUT.CUTOUT.NUM_HOLE_RANGE = (5, 20)
    cfg.INPUT.CUTOUT.RADIUS_RANGE = (0.05, 0.15)
    cfg.INPUT.CUTOUT.COLOR_RANGE = ([0, 255], [0, 255], [0, 255])