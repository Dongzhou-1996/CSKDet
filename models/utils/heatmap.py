import numpy as np
import torch
import torch.nn as nn
import cv2
from typing import Tuple


class KeypointToHeatMap(object):
    def __init__(self,
                 heatmap_hw: Tuple[int, int] = (224 // 4, 224 // 4),
                 gaussian_sigma = None,
                 keypoints_weights=None):
        self.heatmap_hw = heatmap_hw
        self.sigma = gaussian_sigma
        self.kernel_radius = self.sigma * 3
        self.use_kps_weights = keypoints_weights is not None
        self.kps_weights = keypoints_weights

        kernel_size = 2 * self.kernel_radius + 1
        self.kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        x_center = y_center = kernel_size // 2
        for x in range(kernel_size):
            for y in range(kernel_size):
                self.kernel[y, x] = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * self.sigma ** 2))

    def __call__(self, images, targets):
        batch_size = images.shape[0]
        num_kps = targets["keypoints"].shape[1]
        heatmap = np.zeros((batch_size, num_kps, self.heatmap_hw[0], self.heatmap_hw[1]), dtype=np.float32)

        for b in range(batch_size):
            kps = targets["keypoints"][b]
            kps_weights = np.ones((num_kps,), dtype=np.float32)
            if "visible" in targets:
                kps_weights = targets["visible"][b]

            heatmap_kps = (kps / 4 + 0.5).int()

            for kp_id in range(num_kps):
                v = kps_weights[kp_id]
                if v < 0.5:
                    continue

                x, y = heatmap_kps[kp_id]
                ul = [x - self.kernel_radius, y - self.kernel_radius]
                br = [x + self.kernel_radius, y + self.kernel_radius]

                if ul[0] > self.heatmap_hw[1] - 1 or ul[1] > self.heatmap_hw[0] - 1 or br[0] < 0 or br[1] < 0:
                    kps_weights[kp_id] = 0
                    continue

                g_x = (max(0, -ul[0]), min(br[0], self.heatmap_hw[1] - 1) - ul[0])
                g_y = (max(0, -ul[1]), min(br[1], self.heatmap_hw[0] - 1) - ul[1])
                img_x = (max(0, ul[0]), min(br[0], self.heatmap_hw[1] - 1))
                img_y = (max(0, ul[1]), min(br[1], self.heatmap_hw[0] - 1))

                if kps_weights[kp_id] > 0.5:
                    heatmap[b, kp_id, img_y[0]:img_y[1] + 1, img_x[0]:img_x[1] + 1] += self.kernel[g_y[0]:g_y[1] + 1, g_x[0]:g_x[1] + 1]

        targets["heatmap"] = torch.as_tensor(heatmap, dtype=torch.float32)
        return images, targets

if __name__ == '__main__':
    # 设置 batch_size 为 2
    batch_size = 2
    num_keypoints = 5  # 5 个关键点

    # 生成随机关键点坐标 (batch_size, num_keypoints, 2)
    np.random.seed(42)
    keypoints = np.random.randint(0, 224, size=(batch_size, num_keypoints, 2))

    # 设置可见性，前四个关键点可见，最后一个不可见
    visible = np.array([[1, 1, 1, 1, 0],  # 第一个样本
                        [1, 1, 1, 1, 0]], dtype=np.float32)  # 第二个样本

    # 创建示例图像和标签
    images = torch.zeros((batch_size, 3, 224, 224))  # 2 个 3 通道黑色图像
    targets = {
        "keypoints": torch.tensor(keypoints, dtype=torch.float32),
        "visible": torch.tensor(visible, dtype=torch.float32),
    }
    # 实例化转换器并应用
    transform = KeypointToHeatMap(gaussian_sigma=2)
    _, targets = transform(images, targets)

    # 取出生成的热力图
    heatmaps = targets["heatmap"].numpy()

    # 显示结果
    fig, axes = plt.subplots(batch_size, num_keypoints, figsize=(15, 6))

    for b in range(batch_size):
        for i in range(num_keypoints):
            axes[b, i].imshow(heatmaps[b, i], cmap='jet')
            axes[b, i].set_title(f"Sample {b}, KP {i}\nVisible: {int(visible[b, i])}")
            axes[b, i].axis("off")

    plt.show()