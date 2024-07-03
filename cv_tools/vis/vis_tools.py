import os

import cv2
import numpy as np
import torch


from .color import HAND_SKELETON_CMU, HAND_SKELETON_COLOR_CMU

__all__ = ["vis_heatmap", "LandmarkVis", "vis_tensor"]


def vis_tensor(data, max_num=10, prefix="vis_tensor"):
    os.makedirs(prefix, exist_ok=True)
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if data.ndim == 3:
        data = data[None, ...]
    assert data.ndim == 4
    data = data.transpose(0, 2, 3, 1) * 128.0 + 128.0
    data = data.astype(np.uint8)
    for idx in range(min(data.shape[0], max_num)):
        cv2.imwrite(os.path.join(prefix, f"tensor_{idx}.jpg"), data[idx])


def vis_heatmap(heatmap, shape=None, save=False, name="img.jpg"):
    """visualize heatmap or heatmap list"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not isinstance(heatmap, list):
        heatmap = [heatmap]

    num = len(heatmap)
    if shape is None:
        shape = (1, num)
    assert shape[0] * shape[1] == num

    fig = plt.figure()
    for idx, hm in enumerate(heatmap):
        ax = fig.add_subplot(eval(f"{shape[0]}{shape[1]}{idx + 1}"))
        im = ax.imshow(hm, cmap=plt.cm.hot_r)
        plt.colorbar(im)

    if save:
        plt.savefig(name)
    plt.close()


class LandmarkVis:
    def __init__(self, vis_bbox=False, vis_boundary=False, vis_pose=False):
        self._vis_bbox = vis_bbox
        self._vis_boundary = vis_boundary
        self._vis_pose = vis_pose

    def _vis_all(self, img, **kwargs):
        ldmk = kwargs["ldmk"]
        num_ldmk = kwargs["num_ldmk"]
        ldmk_type = kwargs.get("ldmk_type", None)

        if self._vis_bbox:
            bbox = kwargs["bbox"]
            img = self.vis_bbox(img, bbox)
        if self._vis_pose:
            pose = kwargs["head pose"]
            img = self.vis_pose(img, pose)
        if self._vis_boundary:
            img = self.vis_boundary(img, ldmk, num_ldmk, ldmk_type)

        img = self.vis_landmark(img, ldmk, num_ldmk)

    @classmethod
    def vis_landmark(cls, img, ldmk, num_ldmk):
        if isinstance(ldmk, list):
            ldmk = np.array(ldmk)
        ldmk = ldmk.reshape(num_ldmk, -1).round().astype(np.int32)
        for k in range(ldmk.shape[0]):
            img = cv2.circle(
                img,
                (ldmk[k, 0], ldmk[k, 1]),
                color=(0, 255, 255),
                thickness=-1,
                radius=2,
            )
        return img

    @classmethod
    def vis_bbox(cls, img, bbox, color=(0, 255, 0)):
        x1, y1, x2, y2 = [int(x) for x in bbox]
        img = cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color=color,
            thickness=2,
        )
        return img

    @classmethod
    def vis_head_pose(cls, img, pose):
        raise NotImplementedError

    @classmethod
    def vis_boundary(cls, img, ldmk, num_ldmk, ldmk_type="hand21"):
        if ldmk_type.lower() == "hand21":
            if isinstance(ldmk, list):
                ldmk = np.array(ldmk)
            ldmk = ldmk.round().astype(np.int32).reshape(num_ldmk, -1)
            img = draw_hand_skeleton(img, ldmk)
        elif ldmk_type.lower() == "face68":
            raise NotImplementedError()
        elif ldmk_type.lower() == "body17":
            raise NotImplementedError()
        else:
            raise ValueError(f"Not supported ldmk_type: {ldmk_type}")
        return img

    def run(self, img, ldmk):
        self._vis_all(img, ldmk)


def draw_hand_skeleton(
    img,
    kps,
    draw_skeleton=True,
    drawpoint=True,
    pointcolor=(0, 0, 255),
    thickness=1,
):
    kps = kps.round().astype(np.int32)
    if draw_skeleton:
        for j in range(HAND_SKELETON_CMU.shape[0]):
            p1 = HAND_SKELETON_CMU[j, 0]
            p2 = HAND_SKELETON_CMU[j, 1]
            x1 = kps[p1, 0]
            y1 = kps[p1, 1]
            x2 = kps[p2, 0]
            y2 = kps[p2, 1]

            color = HAND_SKELETON_COLOR_CMU[j % len(HAND_SKELETON_COLOR_CMU)]
            cv2.line(
                img,
                (x1, y1),
                (x2, y2),
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

    if drawpoint:
        for pt in kps:
            cv2.circle(img, (pt[0], pt[1]), 1, pointcolor, -1, cv2.LINE_AA)
    return img
