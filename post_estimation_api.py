from flask import Flask, request, jsonify
import tempfile
import os
import torch
import numpy as np
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images

# Initialize Flask app
app = Flask(__name__)

# Device and model setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
model = AsymmetricMASt3R.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


def estimate_rigid_umeyama(A: np.ndarray, B: np.ndarray):
    muA = A.mean(axis=0)
    muB = B.mean(axis=0)
    AA = A - muA
    BB = B - muB
    U, _, Vt = np.linalg.svd(AA.T @ BB)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = muB - R @ muA
    return R, t


@app.route('/estimate', methods=['POST'])
def estimate():
    # Expecting form-data with 'target' and 'current' image files
    if 'target' not in request.files or 'current' not in request.files:
        return jsonify({'error': 'Missing image files'}), 400

    target_file = request.files['target']
    current_file = request.files['current']

    # Save uploaded images to temporary files
    tmp_target = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    tmp_current = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    try:
        tmp_target.write(target_file.read())
        tmp_target.flush()
        tmp_current.write(current_file.read())
        tmp_current.flush()

        # Load and preprocess images
        imgs = load_images([tmp_target.name, tmp_current.name], size=512)
        output = inference([tuple(imgs)], model, DEVICE, batch_size=1, verbose=False)

        # Extract descriptors and 3D points
        desc1_raw = output['pred1']['desc'].squeeze(0)
        _, H, W = desc1_raw.shape
        pts1_raw = output['pred1']['pts3d'].squeeze(0).cpu().numpy()
        pts2_raw = output['pred2']['pts3d_in_other_view'].squeeze(0).cpu().numpy()

        # Prepare matches
        desc1_map = desc1_raw.permute(1, 2, 0).contiguous()
        desc2_map = output['pred2']['desc'].squeeze(0).permute(1, 2, 0).contiguous()
        matches0, matches1 = fast_reciprocal_NNs(desc1_map, desc2_map, subsample_or_initxy1=8, device=DEVICE, dist='dot', block_size=8192)
        matches0 = matches0.cpu().numpy() if torch.is_tensor(matches0) else matches0
        matches1 = matches1.cpu().numpy() if torch.is_tensor(matches1) else matches1

        xs0, ys0 = matches0[:, 0], matches0[:, 1]
        xs1, ys1 = matches1[:, 0], matches1[:, 1]
        valid = (xs0 >= 0) & (xs0 < W) & (ys0 >= 0) & (ys0 < H) & (xs1 >= 0) & (xs1 < W) & (ys1 >= 0) & (ys1 < H)
        idx0 = (ys0[valid] * W + xs0[valid]).astype(int)
        idx1 = (ys1[valid] * W + xs1[valid]).astype(int)

        # Flatten and sample 3D points
        pts1_flat = pts1_raw.reshape(-1, 3) if pts1_raw.ndim == 3 else pts1_raw
        pts2_flat = pts2_raw.reshape(-1, 3) if pts2_raw.ndim == 3 else pts2_raw
        pts1_samples = pts1_flat[idx0]
        pts2_samples = pts2_flat[idx1]

        # Estimate rigid transform
        R, t = estimate_rigid_umeyama(pts1_samples, pts2_samples)

        # Compute planar range and azimuth
        rho_xy = float(np.hypot(t[0], t[2]))
        phi_xy = float(np.arctan2(t[0], t[2]))

        return jsonify({'rho': rho_xy, 'phi': phi_xy})

    finally:
        # Clean up temporary files
        tmp_target.close()
        tmp_current.close()
        os.unlink(tmp_target.name)
        os.unlink(tmp_current.name)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)