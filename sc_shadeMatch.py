import numpy as np
import math
import pandas as pd
import requests
from PIL import Image
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from colormath.color_conversions import convert_color 
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_diff import delta_e_cie2000


def comp_shades():
    '''
    Main worker function:
    - Reads in and constructs database
    - Calls other functions to: 
    1) quantify RGB color from images 
    2) Transfer RGB color into Lab color space
    3) Calculate true distance in Lab color space
    - Saves RGB and Lab colors and Lab color distance to pandas dataframe
    - Returns final data frame
    '''

    # read in info for test case, save to pandas dataframe
    sdat = pd.read_csv('pmg_vendetta_swatch.txt',sep="\t",header=None)
    sdat.columns=['swatch','percent','name']
    ldat = pd.read_csv('pmg_vendetta_lip.txt',header=None)
    ndat = pd.read_csv('pmg_vendetta_nat.txt',header=None)
    sdat['artificial'] = ldat
    sdat['natural'] = ndat
    # 3 types of images: arm swatch, lip application natural and artificial lighting
    usedex = ['swatch','artificial','natural']
    savedex = ['swa','art','nat']
    for j in range(3):
        s_url = list(sdat[usedex[j]])
        # arrays for RGB color, Lab color, and color difference (Lab color space)
        rgb_array = np.full_like(np.zeros((len(s_url),3)),np.nan)
        lab_array = np.full_like(np.zeros((len(s_url),3)),np.nan)
        color_diff = np.full_like(np.zeros(len(s_url)),np.nan)
        for i in range(len(s_url)):
            if i == 0:
                # reference image is the top row in dataframe
                ref_img = load_image(s_url[i])
                ref_color = use_rgbColor(ref_img)
                rgb_array[i] = ref_color
            elif s_url[i][0:4] == 'http':
                # only consider entries with url's
                tmp_img = load_image(s_url[i])
                tmp_color = use_rgbColor(tmp_img)
                tmp_diff,ref_lab,tmp_lab = calc_diff(ref_color,tmp_color)
                color_diff[i] = tmp_diff
                rgb_array[i] = tmp_color
                lab_array[i] = tmp_lab
        lab_array[0] = ref_lab
        # save to data base
        sdat[savedex[j]+' r_rgb'] = rgb_array[:,0]
        sdat[savedex[j]+' g_rgb'] = rgb_array[:,1]
        sdat[savedex[j]+' b_rgb'] = rgb_array[:,2]

        sdat[savedex[j]+' a_lab'] = lab_array[:,0]
        sdat[savedex[j]+' b_lab'] = lab_array[:,1]
        sdat[savedex[j]+' l_lab'] = lab_array[:,2]

        sdat[savedex[j]+' lab_diff'] = color_diff
    return sdat



def load_image(url):
    '''
    Takes in input url, loads image at url, returns the img in an numpy array
    '''
    response = requests.get(url)
    img_src = Image.open(StringIO(response.content))
    img = np.array(img_src.getdata()).reshape(img_src.size[1],img_src.size[0],3)

    return img



def use_rgbColor(img,box_size=180):
    '''
    Takes in input image (numpy array) and calculates mean RGB color for central part of frame (box_size determines width
    and height of area). Returns array of calculated RGB.
    '''
    mid_point = np.array([img.shape[0]/2,img.shape[1]/2])
    use_region = img[mid_point[0]-box_size/2:mid_point[0]+box_size/2,mid_point[1]-box_size/2:mid_point[1]+box_size/2,:]
    use_rgb = np.mean(use_region,axis=(0,1))

    return use_rgb



def calc_diff(rgb_array1,rgb_array2):
    '''
    Translates two input RGB color to Lab color space and calculates distance in Lab color space
    using delte E CIE 2000 equation. Returns color difference and Lab color (array) of two input
    RGB colors.
    '''
    # convert array into RGB color object
    rgb_obj1 = sRGBColor(rgb_array1[0],rgb_array1[1],rgb_array1[2])
    rgb_obj2 = sRGBColor(rgb_array2[0],rgb_array2[1],rgb_array2[2])
    # convert to CIE Lab object
    lab_obj1 = convert_color(rgb_obj1,LabColor)
    lab_obj2 = convert_color(rgb_obj2,LabColor)
    lab1_array = np.array([lab_obj1.lab_a,lab_obj1.lab_b,lab_obj1.lab_l])
    lab2_array = np.array([lab_obj2.lab_a,lab_obj2.lab_b,lab_obj2.lab_l])
    # used delta E CIE 2000 equation to correct for remaining perceptual deformaties
    return delta_e_cie2000(lab_obj1,lab_obj2),lab1_array/100.0,lab2_array/100.0
