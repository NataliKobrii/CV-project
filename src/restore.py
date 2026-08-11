"""Image restoration methods for RQ3.

We model a degraded frame as g = h * f + n, where f is the clean frame, h is
a blur point spread function (PSF) applied with circular convolution, and n
is additive noise. We use circular convolution because the 2D Fourier
transform diagonalizes it exactly, so the measured quality reflects the
restoration method itself rather than the boundary handling.

The frequency-domain methods are the inverse filter, the Wiener filter,
Richardson-Lucy deconvolution, and a Butterworth low-pass filter for
denoising. The spatial baselines are the Gaussian, median, bilateral,
non-local means and unsharp masking filters.

All public functions take and return BGR uint8 images and work in float32
per channel internally. The PSF is assumed to be known, which is the
standard controlled setting. Blind PSF estimation is left as future work and
is discussed in notebook 05.
"""
import cv2
import numpy as np


def _odd(n):
    """Return n rounded up to the nearest odd integer."""
    return int(n) | 1


def motion_psf(length=15, angle_deg=0.0):
    """Return a linear motion-blur PSF: a normalized one-pixel line through
    the center of the kernel."""
    size = _odd(length)
    k = np.zeros((size, size), np.float32)
    c = size // 2
    a = np.deg2rad(angle_deg)
    r = (length - 1) / 2
    p0 = (int(round(c - r * np.cos(a))), int(round(c - r * np.sin(a))))
    p1 = (int(round(c + r * np.cos(a))), int(round(c + r * np.sin(a))))
    cv2.line(k, p0, p1, 1.0, 1)
    return k / k.sum()


def defocus_psf(radius=5):
    """Return a disk-shaped defocus PSF."""
    size = _odd(2 * radius + 1)
    k = np.zeros((size, size), np.float32)
    cv2.circle(k, (size // 2, size // 2), radius, 1.0, -1)
    return k / k.sum()


def _transfer(psf, shape):
    """Return the transfer function H(u, v): the PSF zero-padded to the
    image size, with its center moved to the origin."""
    pad = np.zeros(shape, np.float32)
    kh, kw = psf.shape
    pad[:kh, :kw] = psf
    # Rolling the kernel so its center sits at the origin makes the FFT of
    # the kernel represent a zero-phase (non-shifting) filter.
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.fft.fft2(pad)


def _per_channel(img, fn):
    """Apply a single-channel function to each color channel and clip the
    result to the uint8 range."""
    out = np.stack([fn(img[..., c].astype(np.float32)) for c in range(3)], -1)
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade(img, psf=None, noise_sigma=0.0, sp_amount=0.0, seed=0):
    """Apply blur (circular convolution), Gaussian noise and, optionally,
    salt-and-pepper noise, in that order."""
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32)
    # Blur first and add noise afterwards, following the degradation model.
    if psf is not None:
        H = _transfer(psf, img.shape[:2])
        out = np.stack([np.real(np.fft.ifft2(np.fft.fft2(out[..., c]) * H))
                        for c in range(3)], -1)
    if noise_sigma > 0:
        out = out + rng.normal(0.0, noise_sigma, out.shape)
    out = np.clip(out, 0, 255)
    # Salt-and-pepper noise sets a random subset of pixels to black or white.
    if sp_amount > 0:
        m = rng.random(img.shape[:2])
        out[m < sp_amount / 2] = 0
        out[m > 1 - sp_amount / 2] = 255
    return out.astype(np.uint8)


def inverse_filter(img, psf, eps=1e-3):
    """Apply the pseudo-inverse filter F = H*G / max(|H|^2, eps^2).

    We keep it as the instructive failure case: wherever |H| is close to
    zero, the division amplifies whatever noise is present."""
    H = _transfer(psf, img.shape[:2])
    # The epsilon floor prevents division by values of H that are close to zero.
    denom = np.maximum(np.abs(H) ** 2, eps ** 2)

    def f(ch):
        return np.real(np.fft.ifft2(np.conj(H) * np.fft.fft2(ch) / denom))
    return _per_channel(img, f)


def wiener_filter(img, psf, K=0.01):
    """Apply Wiener deconvolution with a constant noise-to-signal power
    ratio K."""
    H = _transfer(psf, img.shape[:2])
    # The K term suppresses the frequencies where noise dominates the signal.
    W = np.conj(H) / (np.abs(H) ** 2 + K)

    def f(ch):
        return np.real(np.fft.ifft2(W * np.fft.fft2(ch)))
    return _per_channel(img, f)


def wiener_K_for(noise_sigma):
    """Choose K from the known noise level.

    The sweep in scripts/restoration_experiments.py found the best K near
    0.02 at a noise level of sigma 5, and (sigma / 25)^2 is close to
    that value."""
    return float(np.clip((noise_sigma / 25.0) ** 2, 1e-4, 0.25))


def richardson_lucy(img, psf, iters=20):
    """Apply Richardson-Lucy deconvolution with multiplicative updates
    computed through the FFT."""
    H = _transfer(psf, img.shape[:2])
    Hc = np.conj(H)

    def f(ch):
        g = np.maximum(ch, 1e-3) / 255.0
        # Richardson-Lucy starts from a flat gray estimate and refines it.
        est = np.full_like(g, 0.5)
        # Each iteration multiplies the estimate by the back-projected ratio
        # of the observation to the current prediction.
        for _ in range(iters):
            conv = np.real(np.fft.ifft2(np.fft.fft2(est) * H))
            ratio = g / np.maximum(conv, 1e-6)
            est = est * np.real(np.fft.ifft2(np.fft.fft2(ratio) * Hc))
            est = np.clip(est, 0.0, 3.0)
        return est * 255.0
    return _per_channel(img, f)


def butterworth_lowpass(img, cutoff=0.12, order=3):
    """Apply a Butterworth low-pass filter in the frequency domain.

    The cutoff is in cycles per pixel, where 0.5 is the Nyquist frequency.
    This is the frequency-domain counterpart of Gaussian smoothing for
    denoising, with a flatter passband."""
    h, w = img.shape[:2]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    # D holds the radial frequency of every FFT bin.
    D = np.sqrt(fx ** 2 + fy ** 2)
    Hlp = 1.0 / (1.0 + (D / cutoff) ** (2 * order))

    def f(ch):
        return np.real(np.fft.ifft2(np.fft.fft2(ch) * Hlp))
    return _per_channel(img, f)


def unsharp_mask(img, sigma=1.5, amount=1.2):
    """Apply unsharp masking, a sharpening baseline that does not need the
    PSF."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def deblur_methods(psf, noise_sigma=0.0):
    """Return a dictionary from method name to restorer for a blur
    degradation.

    The entry 'none' leaves the image degraded. The entry 'gaussian' is a
    control that shows smoothing cannot invert a blur."""
    K = wiener_K_for(max(noise_sigma, 2.0))
    return {
        "none": lambda im: im,
        "inverse": lambda im: inverse_filter(im, psf),
        "wiener": lambda im: wiener_filter(im, psf, K),
        "rl20": lambda im: richardson_lucy(im, psf, 20),
        "unsharp": lambda im: unsharp_mask(im),
        "gaussian": lambda im: cv2.GaussianBlur(im, (0, 0), 1.0),
    }


def denoise_methods(noise_sigma=15.0, salt_pepper=False):
    """Return a dictionary from method name to restorer for a noise
    degradation.

    The entry 'butterworth' is the frequency-domain method; the rest are
    spatial filters. For salt-and-pepper noise the noise sigma is 0, so the
    parameters fall back to an effective sigma that gives the bilateral and
    non-local means filters a fair setting."""
    sig = noise_sigma if noise_sigma > 0 else 20.0
    h = max(3.0, 0.8 * sig)
    return {
        "none": lambda im: im,
        "gaussian": lambda im: cv2.GaussianBlur(im, (0, 0),
                                                max(0.8, sig / 12.0)),
        "median": lambda im: cv2.medianBlur(im, 5 if salt_pepper else 3),
        "bilateral": lambda im: cv2.bilateralFilter(im, 7, 2.5 * sig, 5.0),
        "nlm": lambda im: cv2.fastNlMeansDenoisingColored(im, None, h, h, 7, 21),
        "butterworth": lambda im: butterworth_lowpass(im),
    }
