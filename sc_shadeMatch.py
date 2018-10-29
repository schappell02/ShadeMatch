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
    '''
    sdat = pd.read_csv('pmg_vendetta_swatch.txt',sep="\t",header=None)
    sdat.columns=['swatch','percent','name']
    ldat = pd.read_csv('pmg_vendetta_lip.txt',header=None)
    ndat = pd.read_csv('pmg_vendetta_nat.txt',header=None)
    sdat['artificial'] = ldat
    sdat['natural'] = ndat
    usedex = ['swatch','artificial','natural']
    savedex = ['swa','art','nat']
    for j in range(3):
        s_url = list(sdat[usedex[j]])
        rgb_array = np.full_like(np.zeros((len(s_url),3)),np.nan)
        lab_array = np.full_like(np.zeros((len(s_url),3)),np.nan)
        color_diff = np.full_like(np.zeros(len(s_url)),np.nan)
        for i in range(len(s_url)):
            if i == 0:
                ref_img = load_image(s_url[i])
                ref_color = use_rgbColor(ref_img)
                rgb_array[i] = ref_color
            elif s_url[i][0:4] == 'http':                
                tmp_img = load_image(s_url[i])
                tmp_color = use_rgbColor(tmp_img)
                tmp_diff,ref_lab,tmp_lab = calc_diff(ref_color,tmp_color)
                color_diff[i] = tmp_diff
                rgb_array[i] = tmp_color
                lab_array[i] = tmp_lab
        lab_array[0] = ref_lab

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
    '''
    response = requests.get(url)
    img_src = Image.open(StringIO(response.content))
    img = np.array(img_src.getdata()).reshape(img_src.size[1],img_src.size[0],3)

    return img



def use_rgbColor(img,box_size=180):
    '''
    '''
    mid_point = np.array([img.shape[0]/2,img.shape[1]/2])
    use_region = img[mid_point[0]-box_size/2:mid_point[0]+box_size/2,mid_point[1]-box_size/2:mid_point[1]+box_size/2,:]
    use_rgb = np.mean(use_region,axis=(0,1))

    return use_rgb



def calc_diff(rgb_array1,rgb_array2):
    '''
    '''
    # convert array into RGB color object
    rgb_obj1 = sRGBColor(rgb_array1[0],rgb_array1[1],rgb_array1[2])
    rgb_obj2 = sRGBColor(rgb_array2[0],rgb_array2[1],rgb_array2[2])
    # convert to CIE Lab object
    lab_obj1 = convert_color(rgb_obj1,LabColor)
    lab_obj2 = convert_color(rgb_obj2,LabColor)
    lab1_array = np.array([lab_obj1.lab_a,lab_obj1.lab_b,lab_obj1.lab_l])
    lab2_array = np.array([lab_obj2.lab_a,lab_obj2.lab_b,lab_obj2.lab_l])

    return delta_e_cie2000(lab_obj1,lab_obj2),lab1_array/100.0,lab2_array/100.0
