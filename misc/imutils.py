import random
import numpy as np
import cv2
from PIL import Image
from PIL import ImageFilter
import PIL
import tifffile


def cv_rotate(image, angle, borderValue):
    """
    rot angle,  fill with borderValue
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    if isinstance(borderValue, int):
        values = (borderValue, borderValue, borderValue)
    else:
        values = borderValue
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=values)


def pil_resize(img, size, order):
    if size[0] == img.shape[0] and size[1] == img.shape[1]:
        return img

    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST

    return np.asarray(Image.fromarray(img).resize(size[::-1], resample))


def pil_rescale(img, scale, order):
    height, width = img.shape[:2]
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)


def pil_rotate(img, degree, default_value):
    if isinstance(default_value, tuple):
        values = (default_value[0], default_value[1], default_value[2], 0)
    else:
        values = (default_value, default_value, default_value,0)
    img = Image.fromarray(img)
    if img.mode =='RGB':
        # set img padding == default_value
        img2 = img.convert('RGBA')
        rot = img2.rotate(degree, expand=1)
        fff = Image.new('RGBA', rot.size, values)  # 灰色
        out = Image.composite(rot, fff, rot)
        img = out.convert(img.mode)

    else:
        # set label padding == default_value
        img2 = img.convert('RGBA')
        rot = img2.rotate(degree, expand=1)
        # a white image same size as rotated image
        fff = Image.new('RGBA', rot.size, values)
        # create a composite image using the alpha layer of rot as a mask
        out = Image.composite(rot, fff, rot)
        img = out.convert(img.mode)

    return np.asarray(img)


def random_resize_long_image_list(img_list, min_long, max_long):
    target_long = random.randint(min_long, max_long)
    h, w = img_list[0].shape[:2]
    if w < h:
        scale = target_long / h
    else:
        scale = target_long / w
    out = []
    for img in img_list:
        out.append(pil_rescale(img, scale, 3) )
    return out


def random_resize_long(img, min_long, max_long):
    target_long = random.randint(min_long, max_long)
    h, w = img.shape[:2]

    if w < h:
        scale = target_long / h
    else:
        scale = target_long / w

    return pil_rescale(img, scale, 3)


def random_scale_list(img_list, scale_range, order):
    """
        输入：图像列表
    """
    target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

    if isinstance(img_list, tuple):
        assert img_list.__len__() == 2
        img1 = []
        img2 = []
        for img in img_list[0]:
            img1.append(pil_rescale(img, target_scale, order[0]))
        for img in img_list[1]:
            img2.append(pil_rescale(img, target_scale, order[1]))
        return (img1, img2)
    else:
        out = []
        for img in img_list:
            out.append(pil_rescale(img, target_scale, order))
        return out


def random_scale(img, scale_range, order):

    target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

    if isinstance(img, tuple):
        return (pil_rescale(img[0], target_scale, order[0]), pil_rescale(img[1], target_scale, order[1]))
    else:
        return pil_rescale(img, target_scale, order)


def random_rotate_list(img_list, max_degree, default_values):
    degree = random.random() * max_degree
    if isinstance(img_list, tuple):
        assert img_list.__len__() == 2
        img1 = []
        img2 = []
        for img in img_list[0]:
            assert isinstance(img, np.ndarray)
            img1.append((pil_rotate(img, degree, default_values[0])))
        for img in img_list[1]:
            img2.append((pil_rotate(img, degree, default_values[1])))
        return (img1, img2)
    else:
        out = []
        for img in img_list:
            out.append(pil_rotate(img, degree, default_values))
        return out


def random_rotate(img, max_degree, default_values):
    degree = random.random() * max_degree
    if isinstance(img, tuple):
        return (pil_rotate(img[0], degree, default_values[0]),
                pil_rotate(img[1], degree, default_values[1]))
    else:
        return pil_rotate(img, degree, default_values)


def random_lr_flip_list(img_list):

    if bool(random.getrandbits(1)):
        if isinstance(img_list, tuple):
            assert img_list.__len__()==2
            img1=list((np.fliplr(m) for m in img_list[0]))
            img2=list((np.fliplr(m) for m in img_list[1]))

            return (img1, img2)
        else:
            return list([np.fliplr(m) for m in img_list])
    else:
        return img_list


def random_lr_flip(img):

    if bool(random.getrandbits(1)):
        if isinstance(img, tuple):
            return tuple([np.fliplr(m) for m in img])
        else:
            return np.fliplr(img)
    else:
        return img


def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw


def random_crop_list(images_list, cropsize, default_values):

    if isinstance(images_list, tuple):
        imgsize = images_list[0][0].shape[:2]
    elif isinstance(images_list, list):
        imgsize = images_list[0].shape[:2]
    else:
        raise RuntimeError('do not support the type of image_list')
    if isinstance(default_values, int): default_values = (default_values,)

    box = get_random_crop_box(imgsize, cropsize)
    if isinstance(images_list, tuple):
        assert images_list.__len__()==2
        img1 = []
        img2 = []
        for img in images_list[0]:
            f = default_values[0]
            if len(img.shape) == 3:
                cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
            else:
                cont = np.ones((cropsize, cropsize), img.dtype)*f
            cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
            img1.append(cont)
        for img in images_list[1]:
            f = default_values[1]
            if len(img.shape) == 3:
                cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
            else:
                cont = np.ones((cropsize, cropsize), img.dtype)*f
            cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
            img2.append(cont)
        return (img1, img2)
    else:
        out = []
        for img in images_list:
            f = default_values
            if len(img.shape) == 3:
                cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype) * f
            else:
                cont = np.ones((cropsize, cropsize), img.dtype) * f
            cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
            out.append(cont)
        return out


def random_crop(images, cropsize, default_values):

    if isinstance(images, np.ndarray): images = (images,)
    if isinstance(default_values, int): default_values = (default_values,)

    imgsize = images[0].shape[:2]
    box = get_random_crop_box(imgsize, cropsize)

    new_images = []
    for img, f in zip(images, default_values):

        if len(img.shape) == 3:
            cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
        else:
            cont = np.ones((cropsize, cropsize), img.dtype)*f
        cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
        new_images.append(cont)

    if len(new_images) == 1:
        new_images = new_images[0]

    return new_images


def top_left_crop(img, cropsize, default_value):

    h, w = img.shape[:2]

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    if len(img.shape) == 2:
        container = np.ones((cropsize, cropsize), img.dtype)*default_value
    else:
        container = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value

    container[:ch, :cw] = img[:ch, :cw]

    return container


def center_crop(img, cropsize, default_value=0):

    h, w = img.shape[:2]

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    sh = h - cropsize
    sw = w - cropsize

    if sw > 0:
        cont_left = 0
        img_left = int(round(sw / 2))
    else:
        cont_left = int(round(-sw / 2))
        img_left = 0

    if sh > 0:
        cont_top = 0
        img_top = int(round(sh / 2))
    else:
        cont_top = int(round(-sh / 2))
        img_top = 0

    if len(img.shape) == 2:
        container = np.ones((cropsize, cropsize), img.dtype)*default_value
    else:
        container = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value

    container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        img[img_top:img_top+ch, img_left:img_left+cw]

    return container


def HWC_to_CHW(img):
    return np.transpose(img, (2, 0, 1))


def pil_blur(img, radius):
    return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=radius)))


def random_blur(img):
    radius = random.random()
    # print('add blur: ', radius)
    if isinstance(img, list):
        out = []
        for im in img:
            out.append(pil_blur(im, radius))
        return out
    elif isinstance(img, np.ndarray):
        return pil_blur(img, radius)
    else:
        print(img)
        raise RuntimeError("do not support the input image type!")


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(np.array(image_numpy,dtype=np.uint8))
    image_pil.save(image_path)


def im2arr(img_path, mode=1, dtype=np.uint8):
    """
    :param img_path:
    :param mode:
    :return: numpy.ndarray, shape: H*W*C
    """
    if mode==1:
        img = PIL.Image.open(img_path)
        arr = np.asarray(img, dtype=dtype)
    else:
        arr = tifffile.imread(img_path)
        if arr.ndim == 3:
            a, b, c = arr.shape
            if a < b and a < c:  # 当arr为C*H*W时，需要交换通道顺序
                arr = arr.transpose([1,2,0])
    # print('shape: ', arr.shape, 'dytpe: ',arr.dtype)
    return arr







