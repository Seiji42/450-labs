from PIL import Image
import argparse
import matplotlib.pyplot as plot
import matplotlib.image as mpimg
import numpy as np
import scipy.misc as scimisc
import scipy.signal as signal
import math

def greyscale(image):
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    gray = np.floor(0.299 * r + 0.587 * g + 0.114 * b)

    return gray.astype(np.uint8)

def brightness(image, value):
    return np.where(255 - value <= image, 255, np.where(0 - value >= image, 0, image + value))

def blur(image):
    filter = np.array([[1,1,1],[1,1,1],[1,1,1]])
    blurred = signal.convolve2d(image, filter, 'same')
    return blurred

def median(image):
    median = np.empty_like(image)

    h, w = median.shape
    for x in range(0, h):
        for y in range (0,w):
            if x == 0 or y == 0 or x == h - 1 or y == w - 1:
                median[x,y] = image[x,y]
            else:
                median[x,y] = 0
                window = image[x-1:x+2,y-1:y+2]
                window = window.flatten()
                window = np.sort(window)
                median[x,y] = window[4] # median
    return median

def sharpen(image):
    filter = np.array([[0,-1,0],[-1,8,-1],[0,-1,0]])
    unsharpen = signal.convolve2d(image, filter, 'same')

    return image + ((1/4) * unsharpen)


def sobel(image):
    filter_x = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    filter_y = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    sobel_x = signal.convolve2d(image, filter_x, 'same')
    sobel_y = signal.convolve2d(image, filter_y, 'same')
    return np.sqrt(np.power(sobel_x, 2) + np.power(sobel_y, 2))

def histogram(image):
    num_pix = 256
    hist = np.histogram(image, num_pix)
    buckets = hist[0]
    total_pix = sum(buckets)
    sum_buckets = np.empty_like(buckets)
    sum_buckets[0] = buckets[0]
    for k in range(1,num_pix):
        sum_buckets[k] = sum_buckets[k - 1] + buckets[k]
    adjusted = np.empty_like(sum_buckets)
    for r in range(0, num_pix):
        adjusted[r] = math.floor(sum_buckets[r] * r / total_pix)
  
    hist_eq = np.empty_like(image)
    h, w = hist_eq.shape
    for x in range(0, h):
        for y in range (0,w):
            val = image[x,y]
            hist_eq[x,y] = adjusted[val]
    return hist_eq

def update_image_values(image):
    updated = np.empty_like(image, dtype=np.uint8)
    updated = np.floor(image * 255)
    return updated.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument('-i', '--image', required=True,
        help='path to the input image')
    ap.add_argument('-g', '--grayscale', action='store_const', const=True, default=False, required=False,
        help='change image to grayscale')
    ap.add_argument('-b', '--brightness', required=False, default = 0, type=int, choices=range(-255,256),
        help='integer value between -255 and 255 to adjust brightness')
    ap.add_argument('-u', '--blur', action='store_const', const=True, default=False, required=False,
        help='perform a uniform blur on image')
    ap.add_argument('-m', '--median', action='store_const', const=True, default=False, required=False,
        help='perform median filter on image')
    ap.add_argument('-s', '--sharp', action='store_const', const=True, default=False, required=False,
        help='sharpen image')
    ap.add_argument('-e', '--edge', action='store_const', const=True, default=False, required=False,
        help='perform sobel edge detection on image')
    ap.add_argument('-hi', '--histogram', action='store_const', const=True, default=False, required=False,
        help='perform histogram equalization on image')
    args = vars(ap.parse_args())
    img_name = args['image'].rsplit('.',1)[0]

    img = mpimg.imread(args['image'])

    if img.dtype == np.float32:
        img = np.floor(img * 255).astype(np.uint8)

    if args['grayscale']:
        img_name += '_gray'
        img = greyscale(img)
    
    img = brightness(img, args['brightness'])
    img_name += '_brightness'

    if args['blur']:
        img_name += '_u_blur'
        img = blur(img)

    if args['median']:
        img_name += '_m_blur'
        img = median(img)

    if args['sharp']:
        img_name += '_sharpen'
        img = sharpen(img)

    if args['edge']:
        img_name += '_sobel'
        img = sobel(img)

    if args['histogram']:
        img_name += '_hist_eq'
        img = histogram(img)
    
    img_name += '.png'
    print(img_name)
    scimisc.imsave(img_name, img)

    imgplot = plot.imshow(img, cmap = plot.get_cmap('gray'))
    plot.show()
    

if __name__ == '__main__':
    main()