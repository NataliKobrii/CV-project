"""Local features and matching for RQ4: SIFT and ORB against a small
learned patch descriptor.

To compare descriptors fairly, we keep the detector fixed: TinyDescNet
reuses the SIFT keypoints and only replaces the descriptor. TinyDescNet is a
shallow convolutional network in the style of L2-Net that maps a 32x32
grayscale patch to a 128-dimensional unit vector.

The descriptor is trained with a triplet loss in the style of HardNet, where
the hardest negative inside the batch completes each triplet. The training
pairs are self-supervised: we warp an aerial frame with a random homography,
jitter its brightness and contrast, and take the patch at the same keypoint
in both views. No labeled correspondences are needed, so the descriptor
trains locally in about a minute. Notebook 06 retrains and evaluates it on
the HPatches benchmark.
"""
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch import nn

sys.path.append(str(Path(__file__).resolve().parent))
from config import DESC_DIM, MODELS_DIR, PATCH_SIZE  # noqa: E402


def as_gray(img):
    """Return the grayscale version of an image, converting from BGR when
    needed."""
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


class ClassicalFeatures:
    """A wrapper around SIFT or ORB. detect_describe(img) returns the
    keypoint coordinates [N, 2], the descriptors [N, D] and the cv2
    keypoints."""

    def __init__(self, kind="sift", n_features=2000):
        self.kind = kind
        if kind == "sift":
            self._f = cv2.SIFT_create(nfeatures=n_features)
            self.norm = cv2.NORM_L2
        elif kind == "orb":
            self._f = cv2.ORB_create(nfeatures=n_features, fastThreshold=10)
            self.norm = cv2.NORM_HAMMING
        else:
            raise ValueError(f"Unknown feature kind {kind!r}")

    def detect_describe(self, img):
        """Detect keypoints and compute their descriptors on one image."""
        kps, desc = self._f.detectAndCompute(as_gray(img), None)
        if desc is None or len(kps) == 0:
            d = 128 if self.kind == "sift" else 32
            dtype = np.float32 if self.kind == "sift" else np.uint8
            return np.zeros((0, 2), np.float32), np.zeros((0, d), dtype), []
        xy = np.array([k.pt for k in kps], np.float32).reshape(-1, 2)
        return xy, desc, list(kps)


class TinyDescNet(nn.Module):
    """A shallow patch descriptor in the style of L2-Net. It maps a 32x32
    grayscale patch to a unit vector of dimension DESC_DIM."""

    def __init__(self, dim=DESC_DIM):
        super().__init__()

        def block(ci, co, stride=1):
            return [nn.Conv2d(ci, co, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(
        # A compact fully convolutional stack: two 3x3 blocks, then two
        # downsampling stages, ending in a 1x1 convolution that outputs the
        # descriptor. This is the L2-Net design at a reduced width.
            *block(1, 32), *block(32, 32),
            *block(32, 64, 2), *block(64, 64),
            *block(64, 128, 2), *block(128, 128),
            nn.Conv2d(128, dim, 8, bias=False),   # Collapses the 8x8 map to 1x1.
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        """Map normalized patches to unit-length descriptors."""
        return nn.functional.normalize(self.net(x).flatten(1), dim=1)


def extract_patches(img, kps, size=PATCH_SIZE, support=2.5):
    """Cut a scale- and orientation-normalized gray patch around every
    keypoint.

    The window spans `support` times kp.size pixels, is rotated to the
    keypoint angle, and is resampled to size x size. Returns
    [N, size, size] float32 in [0, 1]."""
    gray = as_gray(img)
    out = np.zeros((len(kps), size, size), np.float32)
    for i, kp in enumerate(kps):
        # The measurement region scales with the keypoint size, so the same
        # physical area is sampled regardless of the feature's scale.
        span = max(kp.size, 4.0) * support
        # One affine transform rotates, scales and centers the patch.
        M = cv2.getRotationMatrix2D(kp.pt, kp.angle, size / span)
        M[0, 2] += size / 2 - kp.pt[0]
        M[1, 2] += size / 2 - kp.pt[1]
        out[i] = cv2.warpAffine(gray, M, (size, size), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT101)
    return out / 255.0


def _norm_patches(t):
    """Normalize each patch to zero mean and unit standard deviation."""
    return (t - t.mean(dim=(2, 3), keepdim=True)) / (
        t.std(dim=(2, 3), keepdim=True) + 1e-6)


def describe_patches(net, patches, device="cpu", batch=512):
    """Compute descriptors for an array of patches in batches."""
    if len(patches) == 0:
        return np.zeros((0, DESC_DIM), np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(patches), batch):
            p = torch.from_numpy(patches[i:i + batch])[:, None]
            out.append(net(_norm_patches(p).to(device)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


class LearnedFeatures:
    """SIFT keypoints combined with TinyDescNet descriptors."""

    def __init__(self, weights=MODELS_DIR / "tiny_desc.pt", device="cpu",
                 n_features=2000):
        self.det = cv2.SIFT_create(nfeatures=n_features)
        self.net = TinyDescNet()
        if Path(weights).exists():
            self.net.load_state_dict(torch.load(weights, map_location="cpu"))
        else:
            print(f"[match] {weights} not found; TinyDescNet is untrained")
        self.net.eval().to(device)
        self.device = device
        self.norm = cv2.NORM_L2

    def detect_describe(self, img):
        """Detect SIFT keypoints and describe them with TinyDescNet."""
        gray = as_gray(img)
        kps = self.det.detect(gray, None)
        if not kps:
            return np.zeros((0, 2), np.float32), np.zeros((0, DESC_DIM),
                                                          np.float32), []
        desc = describe_patches(self.net, extract_patches(gray, kps),
                                self.device)
        xy = np.array([k.pt for k in kps], np.float32).reshape(-1, 2)
        return xy, desc, list(kps)


def build_features(kind, device="cpu", n_features=2000, weights=None):
    """Construct a feature extractor by name: 'sift', 'orb' or 'tinydesc'."""
    if kind in ("sift", "orb"):
        return ClassicalFeatures(kind, n_features)
    if kind == "tinydesc":
        return LearnedFeatures(weights or MODELS_DIR / "tiny_desc.pt",
                               device, n_features)
    raise ValueError(f"Unknown feature kind {kind!r}")


def match_descriptors(d1, d2, norm=cv2.NORM_L2, ratio=0.8):
    """Match two descriptor sets with the Lowe ratio test. Returns index
    pairs [M, 2] into (d1, d2) and the match distances [M]."""
    if len(d1) < 2 or len(d2) < 2:
        return np.zeros((0, 2), int), np.zeros(0, np.float32)
    # For each descriptor the two nearest neighbors are retrieved, and the
    # ratio test keeps the match only when the best is clearly closer.
    knn = cv2.BFMatcher(norm).knnMatch(d1, d2, k=2)
    idx, dist = [], []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            idx.append((m.queryIdx, m.trainIdx))
            dist.append(m.distance)
    return (np.array(idx, int).reshape(-1, 2),
            np.array(dist, np.float32))


def estimate_homography(p1, p2, thresh=3.0):
    """Estimate a RANSAC homography from p1 to p2. Returns the homography
    (or None) and the inlier mask."""
    if len(p1) < 4:
        return None, np.zeros(len(p1), bool)
    H, mask = cv2.findHomography(p1.astype(np.float64),
                                 p2.astype(np.float64), cv2.RANSAC, thresh)
    inl = mask.ravel().astype(bool) if mask is not None else np.zeros(len(p1),
                                                                      bool)
    return H, inl


def random_homography(shape, rng, max_rot=25.0, max_persp=0.12,
                      scale=(0.7, 1.4), max_trans=0.05):
    """Return a random homography for an image of the given shape, composed
    of rotation, scale, perspective jitter and translation."""
    h, w = shape[:2]
    R = cv2.getRotationMatrix2D((w / 2, h / 2),
                                rng.uniform(-max_rot, max_rot),
                                rng.uniform(*scale))
    H = np.vstack([R, [0.0, 0.0, 1.0]])
    # The perspective part jitters the four corners independently.
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = (src + rng.uniform(-max_persp, max_persp, (4, 2)) * [w, h]
           + rng.uniform(-max_trans, max_trans, 2) * [w, h])
    # Composing a random rotation-scale with a random corner perturbation
    # yields a homography that mimics a viewpoint change.
    P = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    return P @ H


def photometric_jitter(img, rng):
    """Apply a random gamma, contrast and brightness change, sometimes with
    added noise. This is the illumination axis of the benchmark."""
    g = rng.uniform(0.6, 1.6)
    a = rng.uniform(0.7, 1.3)
    b = rng.uniform(-30, 30)
    x = (img.astype(np.float32) / 255.0) ** g * a + b / 255.0
    x = np.clip(x, 0, 1) * 255.0
    if rng.random() < 0.5:
        x = x + rng.normal(0, rng.uniform(2, 8), x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def patch_pairs_from_image(img, rng, n=64, det=None, margin=24):
    """Build positive patch pairs for descriptor training.

    The anchor patch is centered at a SIFT keypoint and the positive at the same
    point in a warped and jittered view. Small orientation and scale
    mismatches are left in deliberately because they act as augmentation.
    Returns (anchors, positives), each of shape [m, S, S]."""
    gray = as_gray(img)
    det = det or cv2.SIFT_create(nfeatures=800)
    kps = det.detect(gray, None)
    if not kps:
        z = np.zeros((0, PATCH_SIZE, PATCH_SIZE), np.float32)
        return z, z
    h, w = gray.shape
    H = random_homography(gray.shape, rng)
    warped = cv2.warpPerspective(photometric_jitter(img, rng), H, (w, h))
    wgray = as_gray(warped)
    scale_f = float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
    rot_deg = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))
    order = rng.permutation(len(kps))
    sel, wkps = [], []
    for j in order:
        kp = kps[j]
        p = cv2.perspectiveTransform(
            np.array([[kp.pt]], np.float64), H).ravel()
        if not (margin < p[0] < w - margin and margin < p[1] < h - margin):
            continue
        sel.append(kp)
        wkps.append(cv2.KeyPoint(float(p[0]), float(p[1]),
                                 kp.size * scale_f,
                                 (kp.angle - rot_deg) % 360.0))
        if len(sel) >= n:
            break
    if not sel:
        z = np.zeros((0, PATCH_SIZE, PATCH_SIZE), np.float32)
        return z, z
    return extract_patches(gray, sel), extract_patches(wgray, wkps)


def train_descriptor(pa, pb, epochs=6, batch=256, lr=1e-3, margin=1.0,
                     device="cpu", seed=0, verbose=True):
    """Train TinyDescNet with the HardNet loss.

    For every positive pair (pa[i], pb[i]), the hardest negative inside the
    batch completes the triplet. Returns the trained network on the CPU in
    eval mode."""
    net = TinyDescNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ta = _norm_patches(torch.from_numpy(pa)[:, None])
    tb = _norm_patches(torch.from_numpy(pb)[:, None])
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(len(ta), generator=g)
        tot, nb = 0.0, 0
        for i in range(0, len(ta) - batch + 1, batch):
            sel = perm[i:i + batch]
            a = net(ta[sel].to(device))
            b = net(tb[sel].to(device))
            # The distance matrix compares every anchor with every positive;
            # its diagonal holds the true pairs.
            d = torch.cdist(a, b)
            eye = torch.eye(len(d), dtype=torch.bool, device=d.device)
            pos = d.diagonal()
            # The hardest negative is the closest non-matching descriptor.
            neg = torch.minimum(d.masked_fill(eye, 9.0).min(1).values,
                                d.masked_fill(eye, 9.0).min(0).values)
            # The triplet margin loss pushes each anchor closer to its true
            # match than to its hardest negative by at least the margin.
            loss = torch.relu(margin + pos - neg).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        if verbose:
            print(f"[desc] Epoch {ep + 1}/{epochs}  loss {tot / max(nb, 1):.4f}")
    return net.cpu().eval()
