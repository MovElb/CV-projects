import numpy as np

def calc_br(img):
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def y_deriv(x, y, image):
    w = image.shape[1]

    if y == 0:
        return image[x][1] - image[x][0]
    elif y == w - 1:
        return image[x][w - 1] - image[x][w - 2]
    else:
        return image[x][y + 1] - image[x][y - 1]


def x_deriv(x, y, image):
    h = image.shape[0]

    if x == 0:
        return image[1][y] - image[0][y]
    elif x == h - 1:
        return image[h - 1][y] - image[h - 2][y]
    else:
        return image[x + 1][y] - image[x - 1][y]


def gradient(brightness_map):
    h, w = brightness_map.shape

    grad = np.zeros(brightness_map.shape)
    for i in range(h):
        for j in range(w):
            dx = x_deriv(i, j, brightness_map)
            dy = y_deriv(i, j, brightness_map)
            grad[i][j] = np.sqrt(dx**2 + dy**2)
    return grad


def min_carve_arr(grad):
    h, w = grad.shape

    c_arr = np.copy(grad)
    for i in range(1, h):
        for j in range(w):
            if j == 0:
                c_arr[i][j] += min(c_arr[i - 1][j], c_arr[i - 1][j + 1])
            elif j == w - 1:
                c_arr[i][j] += min(c_arr[i - 1][j - 1], c_arr[i - 1][j])
            else:
                c_arr[i][j] += min(c_arr[i - 1][j - 1], c_arr[i - 1][j],
                                   c_arr[i - 1][j + 1])
    return c_arr


def argmin(array):
    array = list(array)
    return array.index(min(array))


def carve_mask_and_coord(grad):
    h, w = grad.shape
    c_arr = min_carve_arr(grad)

    last_row_coord = (h - 1, argmin(c_arr[-1]))
    carve_coord = [last_row_coord]

    carv_mask = np.zeros(grad.shape, dtype=np.int)
    carv_mask[h - 1][last_row_coord[1]] = 1

    for i in range(h - 1):
        x, y = carve_coord[0]

        if y == 0:
            min_neighb = argmin(c_arr[x - 1][: 2])
        elif y == w - 1:
            min_neighb = argmin(c_arr[x - 1][w - 2:])
        else:
            min_neighb = argmin(c_arr[x - 1][y - 1: y + 2])

        if y != 0:
            if min_neighb == 0:
                min_neighb = y - 1
            elif min_neighb == 1:
                min_neighb = y
            else:
                min_neighb = y + 1

        carve_coord = [(x - 1, min_neighb)] + carve_coord
        carv_mask[x - 1][min_neighb] = 1

    return carv_mask, carve_coord


def gradient_upper_bound(img):
    return img.shape[0] * img.shape[1] * 256


def apply_mask(energy_map, mask=None):
    if mask is None:
        return energy_map
    e_up_b = gradient_upper_bound(energy_map)
    return energy_map + e_up_b * mask


def remove_carve(img, coord):
    return np.asarray(list(map(lambda row, crd: np.delete(row, crd[1], 0), img, coord)))


def avrg_px(img, coord):
    x, y = coord

    sum = img[x][y].astype(int)
    num = 1
    if y + 1 != img.shape[1]:
        sum += img[x][y + 1].astype(int)
        num += 1
    if y != 0:
        sum += img[x][y - 1].astype(int)
        num += 1
    return sum // num


def add_carve(img, coord):
    return np.asarray(list(map(lambda row, crd:
                               np.insert(row, crd[1] + 1, avrg_px(img, crd), 0),
                               img, coord)))


def seam_carve(img, mode, mask=None):
    """Resizes the image on 1 pixel using seam carving technique.
       Works in 4 modes: vertical(horizontal) shrink(expand). 
       Mask allows to avoid deleting some fragments of picture or
       conversely or force delete them.
       
       Parameters:
       ----------
       img : NumPy 3d-array
           Image for resizing.
       
       mode : string
           String in the following style:
               vertical shrink
               vertical expand
               horizontal shrink
               horizontal shrink
       
       mask : NumPy 2d-array
           Specifies mask for deleting(preserving) fragments of picture.
       
       Returns:
       ----------
       resized_img : NumPy 3d-array
           Resized image
       
       resized_mask : NumPy 2d-array
           Resized mask
       
       carve_mask : NumPy 2d-array
           Binary array where if in x, y is 1 then that pixel should be deleted, 
           if in x, y is 0 then that pixel should be preserved.
    """
    
    if mode[: 4] == 'vert':
        img = np.swapaxes(img, 0, 1)
        if mask is not None:
            mask = np.swapaxes(mask, 0, 1)

    energy_img = calc_br(img)

    carve_mask, coord = carve_mask_and_coord(apply_mask(gradient(energy_img), mask))

    resized_mask = None
    if mode[-6:] == "shrink":
        if mask is not None:
            resized_mask = remove_carve(mask, coord)
        resized_img = remove_carve(img, coord)
    else:
        if mask is not None:
            resized_mask = add_carve(mask, coord)
        resized_img = add_carve(img, coord)

    if mode[: 4] == 'vert':
        carve_mask = np.swapaxes(carve_mask, 0, 1)
        resized_img = np.swapaxes(resized_img, 0, 1)
        if mask is not None:
            resized_mask = np.swapaxes(resized_mask, 0, 1)

    return resized_img, resized_mask, carve_mask
