from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import cv2
from pystackreg import StackReg
from random import randrange


def count_images_per_folder(directory, folders):
    n_images = []
    for folder in folders:
        if folder != '.DS_Store':
            n_images.append(len(os.path.join(directory, folder)))
    
    print('----------------------------------------------------------------------')
    print('There are '+str(len(folders))+' folders, containg an average of '+str(int(np.mean(n_images)))+' images each.')
    print('min and max number of images: ', int(np.min(n_images)), int(np.max(n_images)))
    print('----------------------------------------------------------------------')

def generate_stack(directory):
    print('-------------------------------')
    print('Creation of the stack...')
    names = []
    image_stack = np.zeros((468, 720, 1280), dtype='uint8')
    image_stack_R = np.zeros((468, 720, 1280), dtype='uint8')
    image_stack_G = np.zeros((468, 720, 1280), dtype='uint8')
    image_stack_B  = np.zeros((468, 720, 1280), dtype='uint8')

    vortex_start_date = 20171004
    vortex_end_date = 20200608
    different_view = [20191128]
    
    images = [f for f in os.listdir(os.path.join(directory)) if int(f.split('-')[2]) > vortex_start_date and int(f.split('-')[2]) < vortex_end_date and int(f.split('-')[2]) not in different_view]
    images.sort()
    n = 0
    for image_name in images:
        image = np.asarray(Image.open(os.path.join(directory, image_name)))
        # If the image is not too "white"
        if image.mean() < 131:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            fm = cv2.Laplacian(gray, cv2.CV_64F).var()
            # If the image is not blurry
            if fm > 2000:
                # Add the image into the stack
                image_stack[n] = gray
                image_stack_R[n] = image[:,:,0]
                image_stack_G[n] = image[:,:,1]
                image_stack_B[n] = image[:,:,2]
                names.append(image_name)
                n+=1
    file = open('names.txt','w')
    for item in names:
        file.write(item+"\n")
    file.close()
    try:
        tifffile.imwrite('results/image_stack.tif', image_stack, imagej=True, metadata={'axes': 'TYX'})
        tifffile.imwrite('results/image_stack_R.tif', image_stack_R, imagej=True, metadata={'axes': 'TYX'})
        tifffile.imwrite('results/image_stack_G.tif', image_stack_G, imagej=True, metadata={'axes': 'TYX'})
        tifffile.imwrite('results/image_stack_B.tif', image_stack_B, imagej=True, metadata={'axes': 'TYX'})
    except:
        tifffile.imwrite('results/image_stack.tif', image_stack)
        tifffile.imwrite('results/image_stack_R.tif', image_stack_R)
        tifffile.imwrite('results/image_stack_G.tif', image_stack_G)
        tifffile.imwrite('results/image_stack_B.tif', image_stack_B)
    
    print('Stacks saved')
    print('-------------------------------')
    return image_stack
    
def align_stack(tmats_path = None):
    print('-------------------------------')
    print('Alignment of the stack...')

    image_stack_R = tifffile.imread('results/image_stack_R.tif')
    image_stack_G = tifffile.imread('results/image_stack_G.tif')
    image_stack_B = tifffile.imread('results/image_stack_B.tif')
    image_stack = tifffile.imread('results/image_stack.tif')
    # Align each frame at the previous one
    sr = StackReg(StackReg.TRANSLATION)
    
    # Generate the tmats
    if tmats_path is None:
        # crop image in the corner and align it - apply matrix on the full stack
        #tmats_float = sr.register_stack(image_stack[:, 480:, 960:], reference='previous')
        tmats_float = sr.register_stack(image_stack, reference='previous')
        # Save the transformation matrix into txt
        transformation_matrices = np.zeros((tmats_float.shape[0], 3), dtype=np.int16)
        transformation_matrices[:, 0] = np.arange(1, tmats_float.shape[0]+1)
        transformation_matrices[:, 1:3] = tmats_float[:, 0:2, 2].astype(int)
        np.savetxt('results/transformationMatrix.txt', transformation_matrices, header = 'timePoint, align_t_x, align_t_y', delimiter = ';')
        
    # Upload the tmats
    else:
        tmats_float = np.zeros((image_stack_R.shape[0], 3, 3), dtype='float64')
        with open(tmats_path) as f:
            next(f)
            for i, line in enumerate(f):
                tmats_float[i,0,0] = 1
                tmats_float[i,1,1] = 1
                tmats_float[i,2,2] = 1
                tmats_float[i,0,2] = float(line.split(';')[1][:5])
                tmats_float[i,0,1] = float(line.split(';')[2].replace('\n','')[:5])

    image_stack_aligned_R = np.zeros((468, 720, 1280), dtype='uint8')
    image_stack_aligned_G = np.zeros((468, 720, 1280), dtype='uint8')
    image_stack_aligned_B = np.zeros((468, 720, 1280), dtype='uint8')
    image_stack_aligned_rgb = np.zeros((468, 3, 720, 1280), dtype='uint8')
    
    image_stack_aligned_R = sr.transform_stack(image_stack_R, tmats=tmats_float.astype(int)).astype('uint8')
    image_stack_aligned_G = sr.transform_stack(image_stack_G, tmats=tmats_float.astype(int)).astype('uint8')
    image_stack_aligned_B = sr.transform_stack(image_stack_B, tmats=tmats_float.astype(int)).astype('uint8')
    image_stack_aligned_rgb[:,0,:,:] = image_stack_aligned_R
    image_stack_aligned_rgb[:,1,:,:] = image_stack_aligned_G
    image_stack_aligned_rgb[:,2,:,:] = image_stack_aligned_B
    try:
        """tifffile.imwrite('results/stack_aligned_R.tif', image_stack_aligned_R, imagej=True, metadata={'axes': 'TYX'})
        tifffile.imwrite('results/stack_aligned_G.tif', image_stack_aligned_G, imagej=True, metadata={'axes': 'TYX'})
        tifffile.imwrite('results/stack_aligned_B.tif', image_stack_aligned_B, imagej=True, metadata={'axes': 'TYX'})"""
        tifffile.imwrite('results/stack_aligned_RGB.tif', image_stack_aligned_rgb, imagej=True, metadata={'axes': 'TCYX'})
    except:
        tifffile.imwrite('results/stack_aligned_RGB.tif', image_stack_aligned_rgb)
    print('Aligned stack saved!')
    print('-------------------------------')


def enhance_contrast(image_matrix, bins=256):
    image_flattened = image_matrix.flatten()
    image_hist = np.zeros(bins)

    # frequency count of each pixel
    for pix in image_matrix:
        image_hist[pix] += 1

    # cummulative sum
    cum_sum = np.cumsum(image_hist)
    norm = (cum_sum - cum_sum.min()) * 255
    # normalization of the pixel values
    n_ = cum_sum.max() - cum_sum.min()
    uniform_norm = norm / n_
    uniform_norm = uniform_norm.astype('int')

    # flat histogram
    image_eq = uniform_norm[image_flattened]
    # reshaping the flattened matrix to its original shape
    image_eq = np.reshape(a=image_eq, newshape=image_matrix.shape)

    return image_eq

def equalize_image(image_src, gray_scale=False):
    if not gray_scale:
        r_image = image_src[:, :, 0]
        g_image = image_src[:, :, 1]
        b_image = image_src[:, :, 2]

        r_image_eq = enhance_contrast(image_matrix=r_image)
        g_image_eq = enhance_contrast(image_matrix=g_image)
        b_image_eq = enhance_contrast(image_matrix=b_image)

        image_eq = np.dstack(tup=(r_image_eq, g_image_eq, b_image_eq))
    else:
        image_eq = enhance_contrast(image_matrix=image_src)
    return image_eq

def mosaic():
    print('-------------------------------')
    print('Complete mosaic generation...')
    image_res = np.zeros((720, 1280, 3), dtype='uint8')
    column = 0
    image_stack_aligned = tifffile.imread('results/image_stack_aligned_rgb.tif')
    
    names = open("names.txt", "r").read()
    dates = [n.split('-')[2] for n in names.split("\n")]
    dates = [d[6:8]+'/'+d[4:6]+'/'+d[0:4] for d in dates]
    
    for i in range(256):
        image = image_stack_aligned[i+50]
        image = equalize_image(image)
        # 5 pixel columns for each image in analysis
        for j in range(5):
            image_col = image[:,column+j, :].copy()
            image_res[:,column+j, :] = image_col
            
        column += 5
    # Save result
    tifffile.imwrite('results/mosaic_image.tif', image_res)

    plt.figure()
    plt.imshow(image_res)
    plt.xticks(np.arange(1, 1281, 75), dates[50:306:15], rotation='vertical')
    plt.yticks([])
    plt.savefig('results/mosaic_image_dates.jpg', bbox_inches='tight', dpi=200)

    print('Image saved!')
    print('-------------------------------')

def chessboard():
    print('-------------------------------')
    print('Chessboard image generation...')
    image_res = np.zeros((720, 1280, 3), dtype='uint8')
    image_stack_aligned = tifffile.imread('results/image_stack_aligned_rgb.tif')
    
    """names = open("names.txt", "r").read()
    dates = [n.split('-')[2] for n in names.split("\n")]
    dates = [d[6:8]+'/'+d[4:6]+'/'+d[0:4] for d in dates]"""
    steps = [10, 20, 40, 80]
    for step in steps:
        for column in range(0, 1280, step):
            # 5 pixel columns for each image in analysis
            for row in range(0, 720, step):
                if column < 300: frame = randrange(50, 150)
                if column > 300 and column < 700: frame = randrange(150, 300)
                if column >700: frame = randrange(300, 468)
                image = equalize_image(image_stack_aligned[frame])
                image_res[row:row+step,column:column+step, :] = image[row:row+step,column:column+step, :].copy()
        # Save result
        tifffile.imwrite('results/chessboard_step'+str(step)+'_image.tif', image_res, imagej=True)

    print('Image saved!')
    print('-------------------------------')


def main():
    # TODO: 
    # new alignment -> I tried but result seems not better at all, try again with another corner f the image 
    # checkboard -> did it
    # crop the trees
    # crop the votex
    
    directory = '/Volumes/RECHERCHE/CTR/CI/DCSR/abarenco/impulse/D2c/rawpix/summary_1200/'
    folders = os.listdir(directory)
    folders.sort()
    if not os.path.exists('results/'):
        os.mkdir('results/')
    
    #count_images_per_folder(directory, folders)
    #generate_stack(directory)
    #align_stack('results/transformationMatrix.txt')
    #mosaic()
    chessboard()
    
	
if __name__ == "__main__":
    main()