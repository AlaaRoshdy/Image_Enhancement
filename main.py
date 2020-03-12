import numpy as np
import skimage
from skimage.util import random_noise
import cv2
import helpers as h


############# Median filter testing #############
img = cv2.imread('./assets/original.png', cv2.IMREAD_GRAYSCALE)
img = skimage.img_as_float(img)
ksize = 3

#Add salt and pepper noise
noise = random_noise(img, mode='s&p', clip=False, amount=0.02)

filtered_img = h.median_filter(img,ksize)

h.show_images([img,noise,filtered_img],["original","Noise","Median filtered image"])


############# Negative transform testing #############
img = cv2.imread('./assets/negative.png', cv2.IMREAD_GRAYSCALE)
inv = h.negative_transform(img)
h.show_images([img, inv],["original", "negatively transformed image"])


############# Negative transform testing #############
img = cv2.imread('./assets/bad_kid.tif', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./assets/dark.png', cv2.IMREAD_GRAYSCALE)

img_stretched = h.contrast_stretching(img)
img2_stretched = h.contrast_stretching(img2)
h.show_images([img, img_stretched, img2, img2_stretched],
            ["bad_kid", "bad_kid stretched", "dark", "dark stretched"])


############# Gamma correction testing #############
gamma = 1.5
img_corr = h.gamma_corr(img, gamma)
img2_corr = h.gamma_corr(img2, gamma)
h.show_images([img, img_corr, img2, img2_corr],
            ["bad_kid", "bad_kid gamma", "dark", "dark gamma"])


############# Histogram equalization testing #############
img_eq = h.hist_eq(img)
img2_eq = h.hist_eq(img2)
h.show_images([img, img_eq/255, img2, img2_eq/255],
            ["bad_kid", "bad_kid equalized", "dark", "dark equalized"])