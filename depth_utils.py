"""Utilities for processing depth images.
"""
import numpy as np
from argparse import Namespace

def get_camera_matrix(width, height, fov):
    """Returns a camera matrix from image size and fov."""
    xc = (width - 1.0) / 2.0
    zc = (height - 1.0) / 2.0
    f = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
    camera_matrix = {"xc": xc, "zc": zc, "f": f}
    camera_matrix = Namespace(**camera_matrix)

    return camera_matrix

def get_point_cloud_from_z(Y, camera_matrix, scale=1):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """

    x, z = np.meshgrid(np.arange(Y.shape[-1]), np.arange(Y.shape[-2] - 1, -1, -1))
    for i in range(Y.ndim - 2):
        x = np.expand_dims(x, axis=0)
        z = np.expand_dims(z, axis=0)

    X = (x[::scale, ::scale] - camera_matrix.xc) * Y[::scale, ::scale] / camera_matrix.f
    Z = (z[::scale, ::scale] - camera_matrix.zc) * Y[::scale, ::scale] / camera_matrix.f

    XYZ = np.concatenate(
        (X[..., np.newaxis], Y[::scale, ::scale][..., np.newaxis], Z[..., np.newaxis]),
        axis=X.ndim,
    )

    return XYZ

def normalize(v):
    return v / np.linalg.norm(v)

def get_r_matrix(ax_, angle):
    ax = normalize(ax_)
    if np.abs(angle) > 0.001:
        S_hat = np.array(
            [[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]],
            dtype=np.float32,
        )

        R = (
            np.eye(3)
            + np.sin(angle) * S_hat
            + (1 - np.cos(angle)) * (np.linalg.matrix_power(S_hat, 2))
        )
    else:
        R = np.eye(3)

    return R

def transform_camera_view(XYZ, sensor_height, camera_elevation_degree):
    """
    Transforms the point cloud into geocentric frame to account for
    camera elevation and angle
    Input:
        XYZ                     : ...x3
        sensor_height           : height of the sensor
        camera_elevation_degree : camera elevation to rectify.
    Output:
        XYZ : ...x3
    """

    R = get_r_matrix([1.0, 0.0, 0.0], angle=np.deg2rad(camera_elevation_degree))
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
    XYZ[..., 2] = XYZ[..., 2] + sensor_height

    return XYZ

def transform_pose(XYZ, current_pose):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : camera position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """

    R = get_r_matrix([0.0, 0.0, 1.0], angle=current_pose[2] - np.pi / 2)  
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
    XYZ[:, :, 0] = XYZ[:, :, 0] + current_pose[0]
    XYZ[:, :, 1] = XYZ[:, :, 1] + current_pose[1]

    return XYZ

def bin_points(XYZ_cms, map_size, z_bins, xy_resolution):
    """Bins points into xy-z bins
    XYZ_cms is ... x H x W x3
    Outputs is ... x map_size x map_size x (len(z_bins)+1)
    """
    sh = XYZ_cms.shape
    XYZ_cms = XYZ_cms.reshape([-1, sh[-3], sh[-2], sh[-1]])
    n_z_bins = len(z_bins) + 1
    counts = []
    isvalids = []
    for XYZ_cm in XYZ_cms:
        isnotnan = np.logical_not(np.isnan(XYZ_cm[:, :, 0]))
        X_bin = np.round(XYZ_cm[:, :, 0] / xy_resolution).astype(np.int32)
        Y_bin = np.round(XYZ_cm[:, :, 1] / xy_resolution).astype(np.int32)
        Z_bin = np.digitize(XYZ_cm[:, :, 2], bins=z_bins).astype(np.int32)
               
        isvalid = np.array(
            [
                X_bin >= 0,
                X_bin < map_size,
                Y_bin >= 0,
                Y_bin < map_size,
                Z_bin >= 0,
                Z_bin < n_z_bins,
                isnotnan,
            ]
        )
        isvalid = np.all(isvalid, axis=0)

        ind = (Y_bin * map_size + X_bin) * n_z_bins + Z_bin
        ind[np.logical_not(isvalid)] = 0
        count = np.bincount(
            ind.ravel(),
            isvalid.ravel().astype(np.int32),
            minlength=map_size * map_size * n_z_bins,
        )
        count = np.reshape(count, [map_size, map_size, n_z_bins])
        counts.append(count)
        isvalids.append(isvalid)
    counts = np.array(counts).reshape(list(sh[:-3]) + [map_size, map_size, n_z_bins])
    isvalids = np.array(isvalids).reshape(list(sh[:-3]) + [sh[-3], sh[-2], 1])
    return counts, isvalids