#一次処理を行うための関数
#module import
from SI_function import astro_image
from SI_function import BkgEstimator as BE
import numpy as np
import astropy.io.fits as fits
import re
import os
import tqdm

"""============pathを決めたりディレクトリを作ったり============="""
#パスと出力名を決定
def set_PATH_name(list_list, output_file, type):
    #pathを決める
    r = re.match(r'(.+/)'+type+'.*\.list',list_list)
    PATH = r.groups()[0]
    #out用ディレクトリ作成
    if not os.path.exists(PATH+'out/'):
        os.mkdir(PATH+'out/')
    output_name = PATH +'out/' + output_file

    return PATH, output_name
#まとめて開いて配列に
def open_fits_multi(list_list, PATH):
    images = []
    hdrs = []

    f = open(list_list, 'r')
    for fline in f:
        hdu = fits.open(PATH+fline.rstrip('\n'))
        images.append(hdu[0].data)
        hdrs.append(hdu[0].header)

    return images, hdrs
#ダークをひく(まとめて)
def image_subtract_dark_multi(images, dark_file):
    hdu_Dark = fits.open(dark_file)
    dark = hdu_Dark[0].data

    for i in range(len(images)):
        images[i] = images[i] - dark

    return images

#flatを規格化する (まあ最悪しなくてもいいが...)
def flat_standrize(flat):
    mean = np.median(flat.reshape(-1,1))
    sigma = np.std(flat.reshape(-1,1))

    c_sigma = np.where((mean-sigma*3 < flat) & (flat < mean + sigma*3))
    flat_c = flat[c_sigma]
    return flat/np.max(flat_c.reshape(-1,1))
#ダークとフラットを開く
def open_dark_flat(dark_file, flat_file):
    hdu_dark = fits.open(dark_file)
    dark = hdu_dark[0].data
    hdu_flat = fits.open(flat_file)
    flat = hdu_flat[0].data
    flat = flat_standrize(flat) #規格化
    return dark, flat
#ダーク減算 フラット除算
def image_subtract_dark_divide_flat(image, dark, flat):
    image = image - dark
    image = image/flat
    return image

def image_sky(image, show_image = False):
    #make mask
    mask, mean, median, sigma = BE.make_mask(image, nsigma = 4, show_image = show_image, showtext = False)
    #sky fitting
    image_bkg, image_parameter = BE.sky_fitting(image, mask, nsigma = 4, show_image = show_image, showtext = False)

    return image_bkg

"""===============for multi==============="""
def sigma_clip_average(i, images, n_sigma):
    #i 行を計算しているので
    image_i = []
    #それぞれの i, jについて計算するよ
    for j in range(len(images[0][0])):
        image_i.append(sigma_clip_average_i_j(i, j, images, n_sigma))
    return image_i

def sigma_clip_average_i_j(i, j, images, n_sigma):
    images_i_j = []
    #k マイの画像に対して
    for k in range(len(images)):
        #とりあえず一旦抽出
        images_i_j.append(images[k][i][j])
    #とりあえずiterは1でいいでしょう　大した枚数じゃないし
    #ちゃんとnp.arrayしないとまずい
    median = np.median(np.array(images_i_j))
    sigma = np.std(np.array(images_i_j))
    c_clip = np.where((median-sigma*n_sigma < np.array(images_i_j)) & (np.array(images_i_j) < median + sigma*n_sigma))
    images_i_j_c = np.array(images_i_j)[c_clip]

    return np.mean(images_i_j_c)

def sigma_clip_average_wrapper(args):
    return sigma_clip_average(*args)

"""===============dark===================="""
#ダークを作る
def make_dark(dark_list, output_file):
    PATH, output_name = set_PATH_name(dark_list, output_file, type = 'dark')

    #composit
    data, hdr = average_dark(dark_list, PATH)

    #save
    astro_image.output_fits(data, output_name, header = hdr)

#ダークを平均化する
def average_dark(dark_list, PATH):
    #ファイルオープン
    f = open(dark_list, 'r')
    i=0
    data = 0.0
    #fits読み込み
    for fline in f:
        hdu = fits.open(PATH+fline.rstrip('\n'))
        image = np.float64(hdu[0].data)
        if i == 0:
            hdr = hdu[0].header
        data += image
        i += 1
    return data/float(i), hdr

"""=================flat======================"""
#フラットを作る
def make_flat(flat_list, dark_file, output_file):
    #pathを決める
    PATH, output_name = set_PATH_name(flat_list, output_file, type = 'flat')

    #composit
    data, hdr = average_flat(flat_list, PATH, dark_file)

    #save
    astro_image.output_fits(data, output_name, header = hdr)

#フラットを平均化する
def average_flat(flat_list, PATH, dark_file):
    hdu_Dark = fits.open(dark_file)
    dark = hdu_Dark[0].data
    #ファイルオープン
    f = open(flat_list, 'r')
    i=0
    data = 0.0
    #fits読み込み
    for fline in f:
        hdu = fits.open(PATH+fline.rstrip('\n'))
        image = np.float64(hdu[0].data)-dark
        if i == 0:
            hdr = hdu[0].header
        data += image
        i += 1
    return data/i, hdr

"""===================light====================="""
#一次処理
#ダーク減算　フラット除算
def light_dark_flat(light_list, dark_file, flat_file, sky = False, test = False):
    #darkとflatを開く
    dark, flat = open_dark_flat(dark_file, flat_file)

    PATH, output_name_dummy = set_PATH_name(light_list, 'output_file_dummy', 'light')

    f = open(light_list, 'r')
    for fline in f:
        #出力ファイル名決め
        m = re.match(r'(.+).FIT',fline.rstrip('\n'))
        if sky:
            output_name = PATH +'out/'+ m.groups()[0]+'_dfs.FIT'
        else:
            output_name = PATH +'out/' +m.groups()[0]+'_df.FIT'

        hdu = fits.open(PATH+fline.rstrip('\n'))
        hdr = hdu[0].header
        image = np.float64(hdu[0].data)
        #ダーク減算 フラット除算
        image = image_subtract_dark_divide_flat(image, dark, flat)

        #スカイを3D plane fitするかどうか
        if sky:
            if test:
                image = image_sky(image, show_image = True)
                break
            else:
                image = image_sky(image)

        astro_image.output_fits(image, output_name, header = hdr)
