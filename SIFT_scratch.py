import math
import numpy as np
from scipy import ndimage

# --------------------------------------------------
# 1. Gaussian utilities
# --------------------------------------------------
def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Return a 2-D Gaussian kernel (normalised to 1)."""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)

# --------------------------------------------------
# 2. SIFT class
# --------------------------------------------------
class SIFT:
    def __init__(self,
                 n_octaves: int = 4,
                 n_scales: int = 3,
                 sigma: float = 1.6,
                 contrast_th: float = 0.04,
                 edge_th: float = 10,
                 border: int = 5):
        self.n_oct = n_octaves
        self.n_sc  = n_scales               # scales / octave
        self.sigma = sigma                  # base blur
        self.k     = 2 ** (1 / n_scales)    # scale step
        self.c_th  = contrast_th
        self.e_th  = edge_th
        self.border = border                # ignore border width
        # #images / octave = s+3 → ensures s DoG layers
        self.n_imgs_oct = self.n_sc + 3

    # --------------------------------------------------
    # 2.1 Gaussian pyramid
    # --------------------------------------------------
    def _gaussian_pyramid(self, img_f32):
        pyr = []
        base = img_f32
        for o in range(self.n_oct):
            octave = []
            for s in range(self.n_imgs_oct):
                if o == 0 and s == 0:
                    octave.append(base)
                else:
                    if s == 0:                 # first image of new octave
                        base = pyr[o - 1][self.n_sc]
                        base = base[::2, ::2]  # down-sample by 2
                    sigma = self.sigma * (self.k ** s)
                    ksize = int(2 * np.ceil(2 * sigma) + 1) | 1
                    g = gaussian_kernel(ksize, sigma)
                    octave.append(ndimage.convolve(base, g, mode='nearest'))
                base = octave[-1]
            pyr.append(octave)
        return pyr

    # --------------------------------------------------
    # 2.2 DoG pyramid
    # --------------------------------------------------
    @staticmethod
    def _dog_pyramid(g_pyr):
        dog = []
        for octave in g_pyr:
            dog.append([octave[i + 1] - octave[i] for i in range(len(octave) - 1)])
        return dog

    # --------------------------------------------------
    # 2.3 Extrema test helper
    # --------------------------------------------------
    def _is_extremum(self, dog, o, s, i, j):
        val = dog[o][s][i, j]
        if val == 0:              # quick reject
            return False
        comp = (val > 0)          # compare > for maxima else <
        for ds in (-1, 0, 1):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if ds == di == dj == 0:
                        continue
                    if comp:
                        if dog[o][s + ds][i + di, j + dj] >= val:
                            return False
                    else:
                        if dog[o][s + ds][i + di, j + dj] <= val:
                            return False
        return True

    # --------------------------------------------------
    # 2.4 Hessian-based edge filter
    # --------------------------------------------------
    def _is_edge_like(self, dog_layer, i, j):
        Dxx = dog_layer[i+1, j] + dog_layer[i-1, j] - 2*dog_layer[i, j]
        Dyy = dog_layer[i, j+1] + dog_layer[i, j-1] - 2*dog_layer[i, j]
        Dxy = (dog_layer[i+1, j+1] - dog_layer[i+1, j-1] -
               dog_layer[i-1, j+1] + dog_layer[i-1, j-1]) / 4.0
        tr = Dxx + Dyy
        det = Dxx*Dyy - Dxy**2
        if det <= 0:
            return True
        return (tr**2 / det) > ((self.e_th + 1) ** 2 / self.e_th)

    # --------------------------------------------------
    # 2.5 Sub-pixel quadratic fit (3-D)
    # --------------------------------------------------
    def _localise(self, dog, o, s, i, j, max_iter=5):
        for _ in range(max_iter):
            # 1st derivatives
            Dx = (dog[o][s][i, j+1] - dog[o][s][i, j-1]) / 2.
            Dy = (dog[o][s][i+1, j] - dog[o][s][i-1, j]) / 2.
            Ds = (dog[o][s+1][i, j] - dog[o][s-1][i, j]) / 2.
            # 2nd derivatives
            Dxx = dog[o][s][i, j+1] + dog[o][s][i, j-1] - 2*dog[o][s][i, j]
            Dyy = dog[o][s][i+1, j] + dog[o][s][i-1, j] - 2*dog[o][s][i, j]
            Dss = dog[o][s+1][i, j] + dog[o][s-1][i, j] - 2*dog[o][s][i, j]
            Dxy = (dog[o][s][i+1, j+1] - dog[o][s][i+1, j-1] -
                   dog[o][s][i-1, j+1] + dog[o][s][i-1, j-1]) / 4.
            Dxs = (dog[o][s+1][i, j+1] - dog[o][s+1][i, j-1] -
                   dog[o][s-1][i, j+1] + dog[o][s-1][i, j-1]) / 4.
            Dys = (dog[o][s+1][i+1, j] - dog[o][s+1][i-1, j] -
                   dog[o][s-1][i+1, j] + dog[o][s-1][i-1, j]) / 4.
            J = np.array([Dx, Dy, Ds])
            H = np.array([[Dxx, Dxy, Dxs],
                          [Dxy, Dyy, Dys],
                          [Dxs, Dys, Dss]])
            try:
                offset = -np.linalg.lstsq(H, J, rcond=None)[0]
            except np.linalg.LinAlgError:
                return None
            if np.all(np.abs(offset) < 0.5):
                break
            j += int(round(offset[0]))
            i += int(round(offset[1]))
            s += int(round(offset[2]))
            shape = dog[o][0].shape
            if not (self.border < i < shape[0]-self.border and
                    self.border < j < shape[1]-self.border and
                    1 <= s < self.n_sc+1):
                return None
        contrast = dog[o][s][i, j] + 0.5 * J.dot(offset)
        if abs(contrast) < self.c_th:
            return None
        if self._is_edge_like(dog[o][s], i, j):
            return None
        return (o, s+offset[2], i+offset[1], j+offset[0], contrast)

    # --------------------------------------------------
    # 2.6 Orientation histogram
    # --------------------------------------------------
    @staticmethod
    def _gradients(img):
        dy = ndimage.sobel(img, axis=0, mode='nearest')
        dx = ndimage.sobel(img, axis=1, mode='nearest')
        mag = np.hypot(dx, dy)
        ori = np.arctan2(dy, dx) % (2*np.pi)   # [0,2π)
        return mag, ori

    def _orientations(self, g_pyr, kp,
                      n_bins=36, peak_ratio=0.8, sigma_factor=1.5):
        o, s, y, x, _ = kp
        s_int = int(round(s))
        img = g_pyr[o][s_int]
        mag, ori = self._gradients(img)
        # window (3σ radius in pixels of THIS octave)
        sigma = sigma_factor * (self.k ** s) * (2 ** o)
        r = int(round(3 * sigma))
        hist = np.zeros(n_bins)
        h, w = img.shape
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                yy, xx = int(round(y+i)), int(round(x+j))
                if 0 <= yy < h and 0 <= xx < w:
                    weight = math.exp(-(i*i + j*j)/(2*sigma*sigma))
                    bin_f = (ori[yy, xx] * n_bins) / (2*np.pi)
                    bin_i = int(np.floor(bin_f)) % n_bins
                    hist[bin_i] += weight * mag[yy, xx]
        # smooth + peak picking
        hist = np.convolve(hist, [1/3,1/3,1/3], mode='same')
        maxv = hist.max()
        peaks = np.where((hist > np.roll(hist,  1)) &
                         (hist > np.roll(hist, -1)) &
                         (hist >= peak_ratio*maxv))[0]
        angles = []
        for p in peaks:
            # quadratic interpolation
            l = hist[(p-1) % n_bins]; c = hist[p]; r = hist[(p+1) % n_bins]
            offset = 0.5*(l - r)/(l - 2*c + r)
            angle = 2*np.pi*(p+offset)/n_bins
            angles.append(angle)
        if not angles:                        # always at least the max bin
            angle = 2*np.pi*hist.argmax()/n_bins
            angles = [angle]
        return angles

    # --------------------------------------------------
    # 2.7 Descriptor (4×4×8 = 128)
    # --------------------------------------------------
    def _descriptor(self, g_pyr, kp, angle,
                    n_spatial=4, n_ori=8, window_factor=3.0):
        o, s, y, x, _, angle = kp  # <--- Patch applied here
        s_int = int(round(s))
        img = g_pyr[o][s_int]
        mag, ori = self._gradients(img)
        cos_t, sin_t = math.cos(angle), math.sin(angle)
        bins_per_rad = n_ori / (2*np.pi)
        sigma = window_factor * (self.k ** s) * (2 ** o)
        win_rad = int(round(sigma * (n_spatial+1) / 2))
        hist = np.zeros((n_spatial, n_spatial, n_ori))
        h, w = img.shape
        for i in range(-win_rad, win_rad+1):
            for j in range(-win_rad, win_rad+1):
                yy = int(round(y+i)); xx = int(round(x+j))
                if 0 <= yy < h and 0 <= xx < w:
                    # rotate to kp frame
                    rx = ( cos_t * j + sin_t * i) / sigma
                    ry = (-sin_t * j + cos_t * i) / sigma
                    if abs(rx) > n_spatial/2 or abs(ry) > n_spatial/2:
                        continue
                    # spatial bins
                    sbx = rx + n_spatial/2 - 0.5
                    sby = ry + n_spatial/2 - 0.5
                    ix, iy = int(math.floor(sbx)), int(math.floor(sby))
                    if not (0 <= ix < n_spatial and 0 <= iy < n_spatial):
                        continue
                    # orientation bin
                    theta = (ori[yy, xx] - angle) % (2*np.pi)
                    ob = theta * bins_per_rad
                    io = int(math.floor(ob)) % n_ori
                    # tri-linear interpolation weights
                    wx, wy, wo = sbx-ix, sby-iy, ob-io
                    for dx in (0,1):
                        for dy in (0,1):
                            for do in (0,1):
                                w_tot = ((1-wx) if dx==0 else wx) * \
                                        ((1-wy) if dy==0 else wy) * \
                                        ((1-wo) if do==0 else wo)
                                hist[(iy+dy)%n_spatial,
                                     (ix+dx)%n_spatial,
                                     (io+do)%n_ori] += \
                                     w_tot * mag[yy, xx] * \
                                     math.exp(-(rx*rx+ry*ry)/(2*(0.5*n_spatial)**2))
        vec = hist.flatten()
        # normalise, threshold, renormalise
        vec /= np.linalg.norm(vec) + 1e-7
        vec = np.clip(vec, 0, 0.2)
        vec /= np.linalg.norm(vec) + 1e-7
        return vec.astype(np.float32)

    # --------------------------------------------------
    # 3. Main API: detect + compute
    # --------------------------------------------------
    def detect_and_compute(self, img_u8: np.ndarray):
        """Return keypoints list & 128-D descriptor ndarray."""
        # ensure grayscale float32 [0,1]
        img = img_u8.astype(np.float32) / 255.0

        g_pyr  = self._gaussian_pyramid(img)
        dog_pyr = self._dog_pyramid(g_pyr)

        # ----------- extrema search -------------
        cand = []
        for o in range(self.n_oct):
            for s in range(1, self.n_sc+1):
                dog_layer = dog_pyr[o][s]
                h, w = dog_layer.shape
                for i in range(self.border, h-self.border):
                    for j in range(self.border, w-self.border):
                        if self._is_extremum(dog_pyr, o, s, i, j):
                            cand.append((o, s, i, j))

        # ----------- sub-pixel localisation -------------
        kps = []
        for (o, s, i, j) in cand:
            res = self._localise(dog_pyr, o, s, i, j)
            if res is not None:
                kps.append(res)

        # ----------- orientation assignment -------------
        keypoints = []
        for kp in kps:
            for ang in self._orientations(g_pyr, kp):
                keypoints.append((*kp, ang))   # extend with angle

        # ----------- descriptor extraction -------------
        descs = np.zeros((len(keypoints), 128), dtype=np.float32)
        for idx, kp in enumerate(keypoints):
            descs[idx] = self._descriptor(g_pyr, kp, kp[-1])  # angle is in kp[-1]

        return keypoints, descs

# --------------------------------------------------
# 4. Example usage
# --------------------------------------------------
if __name__ == '__main__':
    import imageio.v3 as iio

    # 4.1 Load & convert to grayscale float32
    img = iio.imread('face.png')         # supply your own image
    if img.ndim == 3:                    # RGB → gray
        img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    img = img.astype(np.uint8)

    # 4.2 Run SIFT
    sift = SIFT(n_octaves=4, n_scales=3,
                contrast_th=0.03, edge_th=10)
    kps, desc = sift.detect_and_compute(img)

    print('Detected keypoints :', len(kps))
    print('Descriptor array    :', desc.shape)
    print('First keypoint info :')
    if kps:
        o, s, y, x, c, ang = kps[0]
        print(f'  octave={o}, scale={s:.2f}, pos=({y:.1f},{x:.1f}), '
              f'contrast={c:.3f}, angle={math.degrees(ang):.1f}°')
        print('  descriptor (first 10 values):', desc[0][:10])
import matplotlib.pyplot as plt

# ... after running sift.detect_and_compute(img)
plt.figure(figsize=(10, 10))
plt.imshow(img, cmap='gray')

# Draw keypoints as circles
for kp in kps:
    o, s, y, x, c, ang = kp
    plt.plot(x, y, 'ro', markersize=5, markerfacecolor='none')  # red circle
plt.title(f'SIFT Keypoints ({len(kps)})')
plt.axis('off')
plt.show()
