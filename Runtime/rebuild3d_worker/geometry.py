import numpy as np


def backproject(depth, intrinsics, extrinsics):
    y, x = np.indices(depth.shape)
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1)
    camera = (pixels @ np.linalg.inv(intrinsics).T) * depth[..., None]
    return (camera - extrinsics[:, 3]) @ extrinsics[:, :3]


def project(points, extrinsic, intrinsic):
    camera = points @ extrinsic[:, :3].T + extrinsic[:, 3]
    projected = camera @ intrinsic.T
    return projected[:, :2] / projected[:, 2, None], camera[:, 2]
