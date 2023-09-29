from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import cv2
from pystackreg import StackReg
from mozaic import create_mozaic
import random


def count_images_per_folder(data_directory):
    """
    Counts the number of images for each folder in the data directory.
    
    Parameters
    ----------
    data_directory : string
        directory in which we have one folder per day,
        eg. /Volumes/RECHERCHE/CTR/CI/DCSR/abarenco/impulse/D2c/rawpix/vortex/
    """
    folders = os.listdir(data_directory)
    folders.sort()
    n_images = []
    for folder in folders:
        if folder != '.DS_Store':
            n_images.append(len(os.path.join(data_directory, folder)))
    
    print('----------------------------------------------------------------------')
    print('There are '+str(len(folders))+' folders, containg an average of '+str(int(np.mean(n_images)))+' images each.')
    print('min and max number of images: ', int(np.min(n_images)), int(np.max(n_images)))
    print('----------------------------------------------------------------------')

def generate_stack(data_directory, vortex_start_date, vortex_end_date, error_dates, results_directory):
    """
    Generate stack of filtered images taken from data_directory and
    the txt file called usable_images_names.txt with their names
    
    Parameters
    ----------
    data_directory : string
        directory in which we have the images
        eg. /Volumes/RECHERCHE/CTR/CI/DCSR/abarenco/impulse/D2c/rawpix/summary_1200/
    vortex_start_date : integer, format yyyymmd
        date to start the stack (reference to info in ImagesDescription.txt)
    vortex_end_date : integer, format yyyymmd
        date to end the stack (reference to info in ImagesDescription.txt)
    error_dates : list
        list of integer dates (yyyymmd) in which there are problems inbetween start and end dates
        (reference to info in ImagesDescription.txt)
    results_directory : string
        path where to save results

    """
    print('-------------------------------')
    print('Creation of the stack...')

    max_n_images = len(os.listdir(data_directory))
    images_shape = np.asarray(Image.open(os.path.join(data_directory, os.listdir(data_directory)[0]))).shape
    # for summary_1200 folder: (945 720 1280)
    names = []
    image_stack = np.zeros((max_n_images, images_shape[0], images_shape[1]), dtype='uint8')
    image_stack_rgb = np.zeros((max_n_images, 3, images_shape[0], images_shape[1]), dtype='uint8')
    
    images = [f for f in os.listdir(os.path.join(data_directory)) if not f.startswith('.') and int(f.split('-')[2]) > vortex_start_date and int(f.split('-')[2]) < vortex_end_date and int(f.split('-')[2]) not in error_dates]
    images.sort()
    n = 0
    for image_name in images:
        image = np.asarray(Image.open(os.path.join(data_directory, image_name)))
        # If the image is not too "white"
        if image.mean() < 131:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            fm = cv2.Laplacian(gray, cv2.CV_64F).var()
            # If the image is not blurry
            if fm > 2000:
                # Add the image into the stack
                image_stack[n] = gray
                image_stack_rgb[n,0,:,:] = image[:,:,0]
                image_stack_rgb[n,1,:,:] = image[:,:,1]
                image_stack_rgb[n,2,:,:] = image[:,:,2]
                names.append(image_name)
                n+=1
    file = open('usable_images_names.txt','w')
    for item in names[:-1]:
        file.write(item+"\n")
    file.write(names[-1])
    file.close()

    image_stack = image_stack[:n, :, :]
    image_stack_rgb = image_stack_rgb[:n, :, :]
    try:
        tifffile.imwrite(results_directory+'image_stack.tif', image_stack, imagej=True, metadata={'axes': 'TYX'})
        tifffile.imwrite(results_directory+'image_stack_rgb.tif', image_stack_rgb, imagej=True, metadata={'axes': 'TCYX'})
    except:
        tifffile.imwrite(results_directory+'image_stack.tif', image_stack)
        tifffile.imwrite(results_directory+'image_stack_rgb.tif', image_stack_rgb)
    
    print('Stacks saved')
    print('-------------------------------')
    return image_stack
    
def align_stack(results_directory, tmats_path = None):
    """
    Align the stack prevuosly generated
    
    Parameters
    ----------
    results_directory : string
        path where the stack has been saved
    vortex_start_date : integer, format yyyymmd
        date to start the stack (reference to info in ImagesDescription.txt)
    vortex_end_date : integer, format yyyymmd
        date to end the stack (reference to info in ImagesDescription.txt)
    error_dates : list
        list of integer dates (yyyymmd) in which there are problems inbetween start and end dates
        (reference to info in ImagesDescription.txt)
    results_directory : string
        path where to save results

    """
    print('-------------------------------')
    print('Alignment of the stack...')

    image_stack_rgb = tifffile.imread(results_directory+'image_stack_rgb.tif')
    image_stack = tifffile.imread(results_directory+'image_stack.tif')
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
        np.savetxt(results_directory+'transformationMatrix.txt', transformation_matrices, header = 'timePoint, align_t_x, align_t_y', delimiter = ';')
        
    # Upload the tmats
    else:
        tmats_float = np.zeros((image_stack.shape[0], 3, 3), dtype='float64')
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
    
    image_stack_aligned_R = sr.transform_stack(image_stack_rgb[:, 0, :, :], tmats=tmats_float.astype(int)).astype('uint8')
    image_stack_aligned_G = sr.transform_stack(image_stack_rgb[:, 1, :, :], tmats=tmats_float.astype(int)).astype('uint8')
    image_stack_aligned_B = sr.transform_stack(image_stack_rgb[:, 2, :, :], tmats=tmats_float.astype(int)).astype('uint8')
    image_stack_aligned_rgb[:,0,:,:] = image_stack_aligned_R
    image_stack_aligned_rgb[:,1,:,:] = image_stack_aligned_G
    image_stack_aligned_rgb[:,2,:,:] = image_stack_aligned_B
    try:
        """tifffile.imwrite(results_directory+'stack_aligned_R.tif', image_stack_aligned_R, imagej=True, metadata={'axes': 'TYX'})
        tifffile.imwrite(results_directory+'stack_aligned_G.tif', image_stack_aligned_G, imagej=True, metadata={'axes': 'TYX'})
        tifffile.imwrite(results_directory+'stack_aligned_B.tif', image_stack_aligned_B, imagej=True, metadata={'axes': 'TYX'})"""
        tifffile.imwrite(results_directory+'stack_aligned_RGB.tif', image_stack_aligned_rgb, imagej=True, metadata={'axes': 'TCYX'})
    except:
        tifffile.imwrite(results_directory+'stack_aligned_RGB.tif', image_stack_aligned_rgb)
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

def timelines(results_directory):
    print('-------------------------------')
    print('Complete timelines generation...')
    image_stack_aligned = tifffile.imread(results_directory+'image_stack_aligned_rgb.tif')
    image_res = np.zeros((image_stack_aligned.shape[1], image_stack_aligned.shape[2], 3), dtype='uint8')
    
    names = open("usable_images_names.txt", "r").read()
    dates = [n.split('-')[2] for n in names.split("\n")]
    dates = [d[6:8]+'/'+d[4:6]+'/'+d[0:4] for d in dates]

    column = 0
    for i in range(256):
        image = image_stack_aligned[i+50]
        image = equalize_image(image)
        # 5 pixel columns for each image in analysis
        image_col = image[:,column:column+5, :].copy()
        image_res[:,column:column+5, :] = image_col
        column += 5
    # Save result
    tifffile.imwrite(results_directory+'timelines_image.tif', image_res)

    plt.figure()
    plt.imshow(image_res)
    plt.xticks(np.arange(1, 1281, 75), dates[50:306:15], rotation='vertical')
    plt.yticks([])
    plt.savefig(results_directory+'timelines_with_dates.jpg', bbox_inches='tight', dpi=200)

    print('Image saved!')
    print('-------------------------------')

def timelines_not_aligned(results_directory):
    print('-------------------------------')
    print('Complete timelines not aligned generation...')
    #image_stack= tifffile.imread('/Volumes/RECHERCHE/CTR/CI/DCSR/abarenco/impulse/D2c/rawpix/Arianna_results/image_stack.tif')
    image_stack_R = tifffile.imread('/Users/aravera/Documents/Impulse_Figerletempsquipasse/image_stack_R.tif')
    image_stack_G = tifffile.imread('/Users/aravera/Documents/Impulse_Figerletempsquipasse/image_stack_G.tif')
    image_stack_B = tifffile.imread('/Users/aravera/Documents/Impulse_Figerletempsquipasse/image_stack_B.tif')
    #image_res = np.zeros((image_stack.shape[1], image_stack.shape[2]), dtype='uint8')
    image_res_R = np.zeros((image_stack_R.shape[1], image_stack_R.shape[2]), dtype='uint8')
    image_res_G = np.zeros((image_stack_R.shape[1], image_stack_R.shape[2]), dtype='uint8')
    image_res_B = np.zeros((image_stack_R.shape[1], image_stack_R.shape[2]), dtype='uint8')

    names = open("usable_images_names.txt", "r").read()
    dates = [n.split('-')[2] for n in names.split("\n")]
    dates = [d[6:8]+'/'+d[4:6]+'/'+d[0:4] for d in dates]

    column = 0
    for i in range(256):
        #image = image_stack[i+50]
        image_R = image_stack_R[i+50]
        image_G = image_stack_G[i+50]
        image_B = image_stack_B[i+50]
        #image = equalize_image(image)
        # 5 pixel columns for each image in analysis
        """image_col = image[:,column:column+5].copy()
        image_res[:,column:column+5] = image_col"""
        
        image_col = image_R[:,column:column+5].copy()
        image_res_R[:,column:column+5] = image_col
        
        image_col = image_G[:,column:column+5].copy()
        image_res_G[:,column:column+5] = image_col
        
        image_col = image_B[:,column:column+5].copy()
        image_res_B[:,column:column+5] = image_col
        column += 5
    # Save result
    tifffile.imwrite(results_directory+'timelines_notaligned_R.tif', image_res_R)
    tifffile.imwrite(results_directory+'timelines_notaligned_G.tif', image_res_G)
    tifffile.imwrite(results_directory+'timelines_notaligned_B.tif', image_res_B)

    """plt.figure()
    plt.imshow(image_res)
    plt.xticks(np.arange(1, 1281, 75), dates[50:306:15], rotation='vertical')
    plt.yticks([])
    plt.savefig(results_directory+'timelines_notaligned_with_dates.jpg', bbox_inches='tight', dpi=200)"""

    print('Image saved!')
    print('-------------------------------')

def trees_timelines(results_directory):
    print('-------------------------------')
    print('Trees timelines generation...')
    image_stack_aligned = tifffile.imread('/Users/aravera/Documents/Impulse_Figerletempsquipasse/results/image_stack_aligned_rgb.tif')
    #results_directory+'image_stack_aligned_rgb.tif')
    image_stack_aligned = image_stack_aligned[:, 350:,:600, :]
    image_res = np.zeros((image_stack_aligned.shape[1], 768, 3), dtype='uint8')
    
    names = open("usable_images_names.txt", "r").read()
    dates = [n.split('-')[2] for n in names.split("\n")]
    dates = [d[6:8]+'/'+d[4:6]+'/'+d[0:4] for d in dates]

    column = 0
    column2 = 0
    for i in range(256):
        image = image_stack_aligned[i+50]
        # 5 pixel columns for each image in analysis
        image_col = image[:,column2:column2+3, :].copy()
        image_res[:,column:column+3, :] = image_col
        column += 3
        column2 += 2
    
    image_res1 = cv2.pyrUp(image_res)
    image_res = cv2.pyrUp(image_res1)
    image_res = cv2.blur(image_res, (3, 3)) 
    # Save result
    tifffile.imwrite(results_directory+'timelines_trees.tif', image_res, imagej=True)

    plt.figure()
    plt.imshow(image_res)
    plt.xticks(np.arange(1, image_res.shape[1]+2, 192), dates[50:306:15][:-1], rotation=40)
    plt.yticks([])
    plt.xticks(fontsize=5)
    plt.savefig(results_directory+'timelines_trees_with_dates.jpg', bbox_inches='tight', dpi=200)

    print('Image saved!')
    print('-------------------------------')


def chessboard(results_directory):
    print('-------------------------------')
    print('Chessboard image generation...')
    image_res = np.zeros((720, 1280, 3), dtype='uint8')
    image_stack_aligned = tifffile.imread(results_directory+'image_stack_aligned_rgb.tif')
    
    """names = open("usable_images_names.txt", "r").read()
    dates = [n.split('-')[2] for n in names.split("\n")]
    dates = [d[6:8]+'/'+d[4:6]+'/'+d[0:4] for d in dates]"""
    steps = [20, 40, 80]
    for step in steps:
        for column in range(0, 1280, step):
            # 5 pixel columns for each image in analysis
            for row in range(0, 720, step):
                if column < 700: frame = random.randrange(50, 150)
                if column > 700: frame = random.randrange(150, 468)
                image = equalize_image(image_stack_aligned[frame])
                image_res[row:row+step,column:column+step, :] = image[row:row+step,column:column+step, :].copy()
        # Save result
        tifffile.imwrite(results_directory+'new_chessboard_step'+str(step)+'_image.tif', image_res, imagej=True)

    print('Image saved!')
    print('-------------------------------')


def generate_daily_stack(results_directory, data_directory, date):
    print('-------------------------------')
    print('Creation of the stack...')
    names = []
    image_stack = np.zeros((481, 720, 1280), dtype='uint8')
    image_stack_R = np.zeros((481, 720, 1280), dtype='uint8')
    image_stack_G = np.zeros((481, 720, 1280), dtype='uint8')
    image_stack_B  = np.zeros((481, 720, 1280), dtype='uint8')
    
    images = os.listdir(data_directory)
    images.sort()
    n = 0
    
    for image_name in images[::2]:
        image = np.asarray(Image.open(os.path.join(data_directory, image_name)))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_stack[n] = gray
        image_stack_R[n] = image[:,:,0]
        image_stack_G[n] = image[:,:,1]
        image_stack_B[n] = image[:,:,2]
        names.append(image_name)
        n+=1

    file = open('daily_names_'+date+'.txt','w')
    for item in names[:-1]:
        file.write(item+"\n")
    file.write(names[-1])
    file.close()
    try:
        tifffile.imwrite(results_directory+'daily_'+date+'_stack.tif', image_stack, imagej=True, metadata={'axes': 'TYX'})
        tifffile.imwrite(results_directory+'daily_'+date+'_stack_R.tif', image_stack_R, imagej=True, metadata={'axes': 'TYX'})
        tifffile.imwrite(results_directory+'daily_'+date+'_stack_G.tif', image_stack_G, imagej=True, metadata={'axes': 'TYX'})
        tifffile.imwrite(results_directory+'daily_'+date+'_stack_B.tif', image_stack_B, imagej=True, metadata={'axes': 'TYX'})
    except:
        tifffile.imwrite(results_directory+'daily_'+date+'_stack.tif', image_stack)
        tifffile.imwrite(results_directory+'daily_'+date+'_stack_R.tif', image_stack_R)
        tifffile.imwrite(results_directory+'daily_'+date+'_stack_G.tif', image_stack_G)
        tifffile.imwrite(results_directory+'daily_'+date+'_stack_B.tif', image_stack_B)
    
    print('Stacks saved')
    print('-------------------------------')
    return image_stack
 
def daily_timelines(results_directory):
    """print('-------------------------------')
    print('Daily timelines generation...')
    image_stack_aligned = tifffile.imread(os.path.join(results_directory,'daily_20200319_stack_rgb.tif'))
    # (481, 720, 1280, 3)
    image_res = np.zeros((image_stack_aligned.shape[1], image_stack_aligned.shape[2], image_stack_aligned.shape[3]), dtype='uint8')

    n_columns_per_image = 6
    column = 0
    for i in range(0, image_stack_aligned.shape[0], 2):
        image = image_stack_aligned[i]
        image = equalize_image(image)
        if column+6 < image_stack_aligned.shape[2]:
            # n_columns_per_image pixel columns for each image in analysis
            image_col = image[:,column:column+n_columns_per_image, :].copy()
            image_res[:,column:column+n_columns_per_image, :] = image_col
            column += n_columns_per_image
        else:
            n_columns_per_image = image_stack_aligned.shape[2] - column
            image_col = image[:,column:column+n_columns_per_image, :].copy()
            image_res[:,column:column+n_columns_per_image, :] = image_col
            column += n_columns_per_image
    # Save result
    tifffile.imwrite(results_directory+'timelines_daily.tif', image_res)

    try:
        names = open("daily_names_20200319.txt", "r").read()
        #hour format = 0500030986
        hours = [n.split('-')[3].split('.')[0][:4] for n in names.split("\n")]
        hours = [h[:2]+':'+h[2:] for h in hours]
        plt.figure()
        plt.imshow(image_res)
        plt.xticks(np.arange(1, 1281, 75), hours[0::28], rotation='vertical')
        plt.yticks([])
        plt.savefig(results_directory+'timelines_daily_with_dates.jpg', bbox_inches='tight', dpi=200)
    except Exception as e:
        print(e)"""
    
    print('-------------------------------')
    print('Daily timelines generation...')
    image_stack_aligned = tifffile.imread(os.path.join(results_directory,'daily_20200319_stack_rgb.tif'))
    # (481, 720, 1280, 3)
    image_res = np.zeros((image_stack_aligned.shape[1], image_stack_aligned.shape[2], image_stack_aligned.shape[3]), dtype='uint8')

    # > 06 and < 0640 + 18-19
    n_columns_per_image = 6
    column = 0

    names = open("daily_names_20200319.txt", "r").read()
    #hour format = 0500030986
    hours = [n.split('-')[3].split('.')[0][:4] for n in names.split("\n")]
    
    skip = 1
    for i in range(len(hours)):
        hour = hours[i]
        if (str(hour[:4]) > '0530' and str(hour[:4]) < '0710') or (str(hour[:4]) > '1700' and str(hour[:4]) < '2000'):
            # 46 images = 276 columns
            image = image_stack_aligned[i]
            image = equalize_image(image)
            if column+6 < image_stack_aligned.shape[2]:
                # n_columns_per_image pixel columns for each image in analysis
                image_col = image[:,column:column+n_columns_per_image, :].copy()
                image_res[:,column:column+n_columns_per_image, :] = image_col
                column += n_columns_per_image
            else:
                n_columns_per_image = image_stack_aligned.shape[2] - column
                image_col = image[:,column:column+n_columns_per_image, :].copy()
                image_res[:,column:column+n_columns_per_image, :] = image_col
                column += n_columns_per_image
        else:
            # altre 201 immagini su 435
            if skip == 3:
                skip = 0
                image = image_stack_aligned[i]
                image = equalize_image(image)
                if column+6 < image_stack_aligned.shape[2]:
                    # n_columns_per_image pixel columns for each image in analysis
                    image_col = image[:,column:column+n_columns_per_image, :].copy()
                    image_res[:,column:column+n_columns_per_image, :] = image_col
                    column += n_columns_per_image
                else:
                    n_columns_per_image = image_stack_aligned.shape[2] - column
                    image_col = image[:,column:column+n_columns_per_image, :].copy()
                    image_res[:,column:column+n_columns_per_image, :] = image_col
                    column += n_columns_per_image
            else:
                skip += 1
                
    # Save result
    tifffile.imwrite(results_directory+'timelines_daily.tif', image_res)

    try:
        names = open("daily_names_20200319.txt", "r").read()
        #hour format = 0500030986
        hours = [n.split('-')[3].split('.')[0][:4] for n in names.split("\n")]
        hours = [h[:2]+':'+h[2:] for h in hours]
        plt.figure()
        plt.imshow(image_res)
        plt.xticks(np.arange(1, 1281, 75), hours[0::28], rotation='vertical')
        plt.yticks([])
        plt.savefig(results_directory+'timelines_daily_with_dates.jpg', bbox_inches='tight', dpi=200)
    except Exception as e:
        print(e)

    print('Image saved!')
    print('-------------------------------')


def mozaic(data_directory, results_directory, imagenames_list, num_bins_x, num_bins_y, id_list):
    print('-------------------------------')
    print('Mozaic generation...')

    dimensions = [720, 1280, 3] # [height, width, channels]   
    result_image = np.zeros((dimensions[2], dimensions[0], dimensions[1]), dtype='uint8')
    
    mozaic_data = create_mozaic(dimensions[1], dimensions[0], num_bins_x, num_bins_y, 0, len(imagenames_list), id_list)
    
    unique_id_mozaic = np.unique([int(i) for x,y,i in mozaic_data])
    for date_id in list(unique_id_mozaic):
        image_name = imagenames_list[int(date_id)]
        image = np.asarray(Image.open(os.path.join(data_directory, image_name)))
        found = False
        for el in mozaic_data:
            x_pos, y_pos, id = el
            id = int(id)
            if id == date_id:
                found = True
                result_image[:, y_pos, x_pos] = image[y_pos, x_pos, :]
        if not found:
            print(found, image_name)

    try:
        tifffile.imwrite(results_directory+'mozaic.tif', result_image, imagej=True, metadata={'axes':'CYX'})
    except:
        tifffile.imwrite(results_directory+'mozaic.tif', result_image)
    
    result_image = result_image.transpose(1, 2, 0)
    im = Image.fromarray(result_image)
    plt.imshow(im)
    plt.show()
    im.save(results_directory+'mozaic.jpg')
    print('Mozaic saved')
    print('-------------------------------')


def main():
    #results_directory = '/Volumes/RECHERCHE/CTR/CI/DCSR/abarenco/impulse/D2c/rawpix/Arianna_results/'
    results_directory = '/Users/aravera/Documents/Impulse_Figerletempsquipasse/'
    if not os.path.exists(results_directory+''):  os.mkdir(results_directory+'')
    
    #data_directory = '/Volumes/RECHERCHE/CTR/CI/DCSR/abarenco/impulse/D2c/rawpix/vortex/'
    #count_images_per_folder(data_directory)
    
    data_directory = '/Volumes/RECHERCHE/CTR/CI/DCSR/abarenco/impulse/D2c/rawpix/summary_1200/'
    
    vortex_start_date = 20171004
    vortex_end_date = 20200608
    error_dates = [20191128]
    #generate_stack(data_directory, vortex_start_date, vortex_end_date, error_dates, results_directory)
    
    #align_stack(results_directory, results_directory+'transformationMatrix.txt')
    #timelines(results_directory)
    #chessboard(results_directory)
    #trees_timelines(results_directory)
    #generate_daily_stack(results_directory, '/Volumes/RECHERCHE/CTR/CI/DCSR/abarenco/impulse/D2c/rawpix/vortex/2020-03-19', '20200319')
    #timelines_not_aligned(results_directory)
    #daily_timelines(results_directory)

    # List of available images
    usable_images_names = open("usable_images_names.txt", "r").read()
    usable_images_names = usable_images_names.split("\n")
    # List of available images dates - yyyymmdd
    usable_images_dates = [d.split('-')[2][0:4]+d.split('-')[2][4:6]+d.split('-')[2][6:8] for d in usable_images_names]

    # Choosing random dates as starting points for "id_list"
    choosen_dates = random.sample(usable_images_dates, 4)
    # If instead you want to choose predefined dates: 
    # check in the "name.txt" file that the date is in the list of available ones!
    # format: yyyymmdd
    choosen_dates = ['20171005', '20180213', '20190522', '20200605']

    # In date_list (x, y) positions must be values within [0, 1] x [0, 1] and they should NOT be collinear
    id_list = [(0.01, 0.01, usable_images_dates.index(choosen_dates[0])), 
                 (0.01, 0.99, usable_images_dates.index(choosen_dates[1])), 
                 (0.99,0.01, usable_images_dates.index(choosen_dates[2])), 
                 (0.99, 0.99, usable_images_dates.index(choosen_dates[3]))]
    # Number of horizontal "boxes" in the output mozaic image
    num_bins_x = 20 
    # Number of vertical "boxes" in the output mozaic image
    num_bins_y = 20
    # Create and save mozaic image
    mozaic(data_directory, results_directory, usable_images_names, num_bins_x, num_bins_y, id_list)
    
    
	
if __name__ == "__main__":
    main()