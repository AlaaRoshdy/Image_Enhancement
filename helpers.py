import numpy as np
import cv2
from skimage import io, img_as_ubyte

def show_images(images, titles):
    """
    Args:
      images (np array) -> array of images
      titles (np array) -> np array of titles
    """
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    assert len(images) == len(titles)
    for title in titles:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    
    for title, img in zip(titles, images):
        cv2.imshow(title, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# How to use show_images([list of images], [list of titles]) They must have the same length
# show_images([img1, img2], ['This is image 1', 'This is image 2'])


def median_filter(img,ksize):
    """
    Args:
      img (np array) -> image as an np array
      ksize (int) -> kernel size of median filter
    Returns:
      filtered_img (np array) -> median filtered image 
    """
    pad = int((ksize-1)/2)
    filtered_img = []
    
    #Pad the image with zeros on all sides of size ksize
    
    # Create a new array of zeros
    padded_img = np.zeros((img.shape[0]+pad*2,img.shape[1]+pad*2))
    # Leave the padded area black and center the image
    padded_img[pad:img.shape[0]+pad,pad:img.shape[1]+pad] = img

    # Perform median filtering
    for i in range(len(padded_img)):
        for j in range(len(padded_img[0])):
            filtered_img.append(np.median(padded_img[i:ksize+i,j:ksize+j]))

    filtered_img = np.asarray(filtered_img).reshape((padded_img.shape[0],padded_img.shape[1]))
    return filtered_img
  
def negative_transform(img):
    """
    Args:
      img (np array) -> image as an np array
    Returns:
      transformed_img -> negatively transformed image as np array
    """
    return np.max(img)-img
  
def contrast_stretching(img):
    """
    Args:
      img (np array) -> image as an np array
    Returns:
      contrast stretched -> contrast stretched image
    """
    return (img-np.min(img))/(np.max(img)-np.min(img))
  
def gamma_corr(img, gamma):
    """
    Args:
      img (np array) -> image as an np array
    Returns:
      gamma_corrected -> gamma corrected image
    """
    return ((img/255)**gamma)
  
def hist_eq(img):
    """
    Args:
      img (np array) -> image as an np array
    Returns:
      equalized_img -> equalized image
    """
    # Flatten the image into 1 dimension: pixels
    pixels = img.flatten()

    # Generate a cumulative histogram
    hist, bins = np.histogram(img.reshape(-1), bins=256, range = (0,256), normed = True)
    cdf = np.cumsum(hist)
    new_pixels = np.interp(pixels, bins[:-1], cdf*255)

    # Reshape new_pixels as a 2-D array: new_image
    equalized_img = new_pixels.reshape((img.shape[0],img.shape[1]))
    
    return equalized_img
  
def save_image(path, im):
  return io.imsave(path, img_as_ubyte(im.copy()))
