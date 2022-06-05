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

#まとめて開いて配列に渡す モノクロ画像
def open_fits_multi(list_list, PATH):
    images = []
    hdrs = []

    f = open(list_list, 'r')
    for fline in f:
        hdu = fits.open(PATH+fline.rstrip('\n'))
        images.append(hdu[0].data)
        hdrs.append(hdu[0].header)

    return images, hdrs

#ダーク減算(まとめて) フラット画像に適用
def image_subtract_dark_multi(images, dark_file):
    hdu_Dark = fits.open(dark_file)
    dark = hdu_Dark[0].data

    for i in range(len(images)):
        images[i] = images[i] - dark

    return images

#flatの規格化 (まあ最悪しなくてもいいが...)
def flat_standrize(flat):
    #統計量を求めて
    mean = np.median(flat.reshape(-1,1))
    sigma = np.std(flat.reshape(-1,1))

    #シグマクリップをして最大値を取得 おそらく画面中央部に相当
    c_sigma = np.where((mean-sigma*3 < flat) & (flat < mean + sigma*3))
    flat_c = flat[c_sigma]

    #中央が1になるように
    return flat/np.max(flat_c.reshape(-1,1))

#ダークとフラットを開く ライト画像に適用
def open_dark_flat(dark_file, flat_file):
    #画像を開いて
    hdu_dark = fits.open(dark_file)
    dark = hdu_dark[0].data
    hdu_flat = fits.open(flat_file)
    flat = hdu_flat[0].data
    flat = flat_standrize(flat) #規格化

    #返す
    return dark, flat

#ダーク減算 フラット除算
def image_subtract_dark_divide_flat(image, dark, flat):
    image = image - dark
    image = image/flat
    return image

#sky fit
def image_sky(image, show_image = False):
    #make mask
    mask, mean, median, sigma = BE.make_mask(image, nsigma = 4, show_image = show_image, showtext = False)
    #sky fitting
    image_bkg, image_parameter = BE.sky_fitting(image, mask, nsigma = 4, show_image = show_image, showtext = False)
    #改めてスカイの統計量を測定
    mask, mean, median, sigma = BE.make_mask(image_bkg, nsigma = 4, show_image = False, showtext = False)
    return image_bkg, mean

#σクリッピングコンポジっとを並列化して行うため
"""===============for multi==============="""
#σクリッピングで平均化 i行目に対して計算
def sigma_clip_average(i, images, n_sigma):
    #i 行を計算しているので
    image_i = []
    #それぞれの i, jについて計算するよ
    for j in range(len(images[0][0])):
        #σクリップする関数を呼び出す
        image_i.append(sigma_clip_average_i_j(i, j, images, n_sigma))

    return image_i

#i, jのデータに対してシグマクリップ平均
def sigma_clip_average_i_j(i, j, images, n_sigma):
    images_i_j = []
    #k マイの画像に対して
    for k in range(len(images)):
        #一旦抽出
        images_i_j.append(images[k][i][j])

    #iterは1回で十分音判断
    #ちゃんとnp.arrayしないとまずい
    median = np.median(np.array(images_i_j))
    sigma = np.std(np.array(images_i_j))
    c_clip = np.where((median-sigma*n_sigma < np.array(images_i_j)) & (np.array(images_i_j) < median + sigma*n_sigma))
    images_i_j_c = np.array(images_i_j)[c_clip]

    return np.mean(images_i_j_c)

#wrapper
def sigma_clip_average_wrapper(args):
    return sigma_clip_average(*args)


"""===============dark===================="""
#ダークを作る
def make_dark(dark_list, output_file):
    #開いて
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
#skyをTrueにすると 二次元平面で近似して減算
#test = Trueなら全て実行せずに一枚目のマスク画像とaky画像を表示
def light_dark_flat(light_list, dark_file, flat_file, sky = False, test = False):
    #darkとflatを開く
    dark, flat = open_dark_flat(dark_file, flat_file)

    #pathの設定
    PATH, output_name_dummy = set_PATH_name(light_list, 'output_file_dummy', 'light')

    f = open(light_list, 'r')
    for fline in f:
        #出力ファイル名決め
        m = re.match(r'(.+).FIT',fline.rstrip('\n'))
        if sky:
            output_name = PATH +'out/'+ m.groups()[0]+'_dfs.FIT'
        else:
            output_name = PATH +'out/' +m.groups()[0]+'_df.FIT'

        #lightを開いて
        hdu = fits.open(PATH+fline.rstrip('\n'))
        hdr = hdu[0].header
        image = np.float64(hdu[0].data)
        #ダーク減算 フラット除算
        image = image_subtract_dark_divide_flat(image, dark, flat)

        #スカイを3D plane fit
        if sky:
            if test:
                image, mean = image_sky(image, show_image = True)
                break
            else:
                image, mean = image_sky(image)

        #スカイの平均値をヘッダーに出力　後の補正で使用
        if sky:
            add_header = ['sky_mean']
            add_header_value = [mean]
            add_header_comment = ['mean value of sky']
            astro_image.output_fits(image, output_name, header = hdr, add_header = add_header, add_header_value = add_header_value, add_header_comment = add_header_comment)
        else:
            astro_image.output_fits(image, output_name, header = hdr)
