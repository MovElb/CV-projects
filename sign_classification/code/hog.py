import numpy as np

from skimage.color import rgb2gray
from skimage.transform import resize

def preprocess_img(img, size=(64, 64), preserve_rgb=False):
    if not preserve_rgb:
        img = rgb2gray(img)

    img = resize(img, size)
    return img


def hog_histograms(magnitude, angles, cellx, celly, n_cellx, n_celly,
                   orientations, orientation_histogram):
    angle_period = 180. / orientations
    for cy_cnt in range(n_celly):
        for cx_cnt in range(n_cellx):
            pnt_x = cellx * cx_cnt
            pnt_y = celly * cy_cnt

            histogram = np.zeros(orientations)
            for y in range(pnt_y, pnt_y + celly):
                for x in range(pnt_x, pnt_x + cellx):
                    _angle = angles[y][x]
                    bin_n = int(_angle // angle_period)
                    relative_angle = _angle % angle_period
                    partition = relative_angle / angle_period

                    histogram[bin_n % orientations] += (1. - partition) * magnitude[y][x]
                    histogram[(bin_n + 1) % orientations] += partition * magnitude[y][x]

            orientation_histogram[cy_cnt][cx_cnt] = histogram

    return orientation_histogram


def extract_hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                norm_eps=1e-12, apply_sqrt=True):            
    """Extracts a HOG vector from given image.
    
       Parameters:
       ----------
       img : NumPy 3d-array
           Initial image.
       
       orientations : integer
           Number of bins.
       
       pixels_per_cell : tuple
           Size of cell window.
       
       cells_per_block : tuple
           Number of cells per block.
       
       norm_eps : double
           Epsilon in normalization formula.
       
       apply_sqrt : bool
           Apply a square root on image.
    """
    
    img = preprocess_img(img)

    if apply_sqrt:
        img = np.sqrt(img)

    dy, dx = np.gradient(img, .5)

    magn = np.hypot(dx, dy)
    angl = np.rad2deg(np.arctan2(dy, dx)) % 180

    sz_y, sz_x = img.shape
    cellx, celly = pixels_per_cell
    blockx, blocky = cells_per_block

    n_cellx = int(sz_x // cellx)
    n_celly = int(sz_y // celly)

    orientation_histogram = np.zeros((n_celly, n_cellx, orientations))
    orientation_histogram = hog_histograms(magn, angl, cellx, celly, n_cellx, n_celly,
                                           orientations, orientation_histogram)

    n_blockx = n_cellx - blockx + 1
    n_blocky = n_celly - blocky + 1

    norm_blocks = np.zeros((n_blocky, n_blockx, blocky, blockx, orientations))

    for y in range(n_blocky):
        for x in range(n_blockx):
            block = orientation_histogram[y: y + blocky, x: x + blockx, :]
            norm_blocks[y, x, :] = block / np.sqrt(np.sum(block ** 2) + norm_eps ** 2)

    return np.ravel(norm_blocks)
