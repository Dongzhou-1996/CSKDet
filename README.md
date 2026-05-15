# CSKDet
Cross-category Spacecraft Keypoints Detection Method with Visual Feature Prompts

Spacecraft visual pose estimation is the technical core of intelligent on-orbit services, often implemented through a two-stage approach that combines keypoint detection and pose solver. However, existing spacecraft keypoint detection methods are typically trained using visual data from a single spacecraft, making them inapplicable to other types of spacecraft targets. This significantly hinders the promotion and application of space on-orbit services. To address this issue, this paper proposes a cross-category spacecraft keypoint detection method based on visual feature prompts, named as CSKDet (cross-category spacecraft keypoints detector). When applied to a new target spacecraft of an unknown category, this method only requires one support image and its corresponding keypoint annotations to accurately predict the positions of the target spacecraft's keypoints in a query image. To further validate the effectiveness of the proposed method, a spacecrafts pose estimation (SPE) dataset, was constructed using a virtual simulation platform. This dataset includes various types of spacecraft, annotated with 2D keypoints and 3D pose labels. Extensive experiments conducted on this dataset demonstrate that the proposed method excels in cross-category spacecraft keypoint detection tasks, significantly outperforming current mainstream keypoint detection approaches. Moreover, when combined with traditional PnP algorithms, this method enables high-precision pose estimation for arbitrary spacecraft. 

## Getting Started
### Conda Environment
We train and evaluate our model on Python 3.10 and Pytorch 2.3.1 with CUDA 11.8.

### SKD Dataset

Please prepare the SPE dataset for training and evaluation.

You can download the SPE dataset [HERE](https://pan.baidu.com/s/1hErUk3jyHgYkgf6LokzBXw?pwd=krm9).
