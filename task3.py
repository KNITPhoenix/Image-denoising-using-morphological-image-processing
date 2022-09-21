import cv2
from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
import numpy as np


def morph_erode(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return erode_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology erosion on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    """

    # TO DO: implement your solution here
    thres = 128
    """img1 = np.where((img > thres), img, 255)
    img2 = np.where((img <= thres), img1, 0)"""  # img2 has the binarised image
    img2=img
    kernel = np.full((3, 3),255)
    r,c=img2.shape
    z = np.zeros((img2.shape[0] + len(kernel) - 1, img2.shape[1] + len(kernel) - 1))
    z[len(kernel) // 2:img2.shape[0] + (len(kernel) // 2), len(kernel) // 2:img2.shape[1] + (len(kernel) // 2)] = img2
    sub=np.array([z[i:(i+3),j:(j+3)] for i in range(r) for j in range(c)])

    erode_img=np.array([255 if (i==kernel).all() else 0 for i in sub])
    erode_img=erode_img.reshape((r,c))
    return erode_img



def morph_dilate(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return dilate_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology dilation on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    """

    # TO DO: implement your solution here
    """thres=128
    img1=np.where((img>thres),img,255)
    img2=np.where((img<=thres),img1,0)"""    #img2 has the binarised image
    img2=img
    kernel=np.full((3,3),255)
    z = np.zeros((img2.shape[0] + len(kernel) - 1, img2.shape[1] + len(kernel) - 1))
    z[len(kernel)//2:img2.shape[0] + (len(kernel) // 2), len(kernel) // 2:img2.shape[1] + (len(kernel) // 2)] = img2
    sub=np.array([z[i:(i+3),j:(j+3)] for i in range(img2.shape[0]) for j in range(img2.shape[1])])

    dilate_img=np.array([255 if (i==kernel).any() else 0 for i in sub])
    dilate_img=dilate_img.reshape((img2.shape[0],img2.shape[1]))

    return dilate_img

def morph_open(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return open_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology opening on input binary image.
    Use 3x3 squared structuring element of all 1's.
    You can use the combination of above morph_erode/dilate functions for this.
    """

    # TO DO: implement your solution here
    erode=morph_erode(img)
    open_img=morph_dilate(erode)
    return open_img


def morph_close(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return close_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology closing on input binary image.
    Use 3x3 squared structuring element of all 1's.
    You can use the combination of above morph_erode/dilate functions for this.
    """

    # TO DO: implement your solution here
    dilate=morph_dilate(img)
    close_img=morph_erode(dilate)
    return close_img


def denoise(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Remove noises from binary image using morphology operations.
    If you convert the dtype of input binary image from int to bool,
    make sure to convert the dtype of returned image back to int.
    """

    # TO DO: implement your solution here
    erode=morph_erode(img)
    open=morph_open(erode)
    dilate=morph_dilate(open)
    denoise_img=morph_close(dilate)
    return denoise_img


def boundary(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Extract boundaries from binary image using morphology operations.
    If you convert the dtype of input binary image from int to bool,
    make sure to convert the dtype of returned image back to int.
    """

    # TO DO: implement your solution here
    bound_img=img-morph_erode(img)
    return bound_img

if __name__ == "__main__":
    img = imread('task3.png', IMREAD_GRAYSCALE)
    denoise_img = denoise(img)
    imwrite('results/task3_denoise.jpg', denoise_img)
    bound_img = boundary(denoise_img)
    imwrite('results/task3_boundary.jpg', bound_img)
