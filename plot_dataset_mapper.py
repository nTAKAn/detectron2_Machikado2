import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_dataset_mapper(dataset_dicts, mapper):
    """
    データマッパーの出力をテスト表示する
    """
    plt.figure(figsize=(16, 24))  # サイズは調整してください

    for i, dataset_dict in enumerate(random.sample(dataset_dicts, 10)):
        data = mapper(dataset_dict)

        img = data['image'].numpy()
        img = img.transpose([1, 2, 0])[:, :, ::-1]

        bboxes = data['instances'].gt_masks.get_bounding_boxes()
        polygons = data['instances'].gt_masks.polygons

        assert len(bboxes), 'bboxがない'
        assert len(polygons), 'マスクがない'

        ax = plt.subplot(5, 2, i + 1)
        ax.imshow(img)

        for bbox in bboxes:
            bbox = bbox.numpy().astype(np.int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            r = patches.Rectangle(xy=(x1, y1), width=(x2 - x1), height=(y2 - y1), ec='r', linewidth=5, fill=False, alpha=0.5)
            ax.add_patch(r)

        for polygon in polygons:
            for _polygon in polygon:
                pt = [(x, y) for x, y in zip(_polygon[::2], _polygon[1::2])]
                ax.add_patch(plt.Polygon(pt, fc='g', alpha=0.5))

        ax.set_title(dataset_dict['file_name'])
        ax.axis('off')
    plt.show()
