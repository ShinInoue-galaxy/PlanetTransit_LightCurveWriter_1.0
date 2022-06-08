#ライトカーブを求めるのに使う関数
#module import
import re
import tqdm
import os
import numpy as np
import image_registration as im_regi # image_registrationパッケージを読み込む
from SI_function import normal_func
from photutils.aperture import CircularAperture
from photutils.aperture.stats import ApertureStats
from photutils.aperture import aperture_photometry
from photutils.aperture import CircularAnnulus
from astropy.stats import SigmaClip

"""=========================クラスで管理================"""
class Circle(object):
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color

    def print_all(self):
        print(self.x, self.y, self.radius, self.color)

class Date(object):
    def __init__(self, y, mo, d, h, m, s):
        self.y = y
        self.mo = mo
        self.d = d
        self.h = h
        self.m = m
        self.s = s

    def show_date(self):
        return (str(self.y)+'/'+str(self.mo)+'/'+str(self.d)+' '+str(self.h)+':'+str(self.m)+':'+str(self.s))

    def return_seconds(self):
        return float(self.d)*24.*3600. + float(self.h)*3600. + float(self.m)*60 + self.s


"""====================-関数================="""
#skyのofsetの処理
def sky_offset(images, hdrs):
    means = []
    for i in range(len(hdrs)):
        means.append(float(hdrs[i]['SKY_MEAN']))
    mean_means = np.mean(np.array(means))

    #offset補正したものを出力
    for i in range(len(images)):
        images[i] = images[i] + mean_means - means[i]
    return images

#reg fileの読み込み
def open_reg_circle(reg_file):
    f = open(reg_file, 'r')
    circles = []
    for fline in f:
        r = re.match(r'circle\((\d+.\d+),(\d+.\d+),(\d+.\d+)\) # color=(\S+)',fline)
        m = re.match(r'circle\((\d+.\d+),(\d+.\d+),(\d+.\d+)\)',fline)
        if r:
            circ = Circle(float(r.group(1)), float(r.group(2)), float(r.group(3)), r.group(4))
            circles.append(circ)
        elif m:
            circ = Circle(float(m.group(1)), float(m.group(2)), float(m.group(3)), 'green')
            circles.append(circ)
    return circles

#regfileで指定した天体の範囲の重心を返す
#天体一つについて
def calc_centroid(image ,displacement ,circle, r_pix = False):
    #extractに用いる半径の決定
    if r_pix:
        r = r_pix
    else:
        r = circle.radius

    xx = 0
    yy = 0
    value_temp = 0.0

    #まず最大値を取る座標を求める。
    for i in range(int(4*r)): #y
        for j in range(int(4*r)):#x
            x = int(circle.x + displacement[0] -2*r) + j
            y = int(circle.y + displacement[1] -2*r) + i
            if value_temp < image[y][x]:
                value_temp = image[y][x]
                xx = x #これは最大値をとるindex (変位を加えたそれぞれのデータに対しての 正確なindex)
                yy = y
    #重心を求める
    sum_value = 0.0
    sum_x_value = 0.0
    sum_y_value = 0.0

    for i in range(int(4*r)): #y
        for j in range(int(4*r)):#x
            x = xx - int(2*r) + j
            y = yy - int(2*r) + i
            if normal_func.dr(x,y,xx,yy) < r:
                sum_value += image[y][x]
                sum_x_value += image[y][x]*x
                sum_y_value +=image[y][x]*y
    x_g = sum_x_value/sum_value
    y_g = sum_y_value/sum_value

    return x_g, y_g, r

#位置ずれを計算。あるいは既に計算済みで貼れば読み込む
def calc_read_displacements(images, displacement_file):
    if os.path.exists(displacement_file):
        print('read existing file: '+displacement_file)
        displacements = np.loadtxt(displacement_file)
    else:
        print('Starting calculation of displacement!')
        image_cri = images[0]
        displacements = [[0.0, 0.0]]
        for i in tqdm.tqdm(range(len(images)-1)):
            displacement = im_regi.cross_correlation_shifts(image_cri, images[i+1])
            #displacement = im_regi.chi2_shift(data_1, data_2)
            #image_tmp = np.roll(images[i+1], (-int(displacement[1]), -int(displacement[0])), axis=(0, 1)) #y, xこれで位置ズレ直せる
            displacements.append([float(displacement[0]),float(displacement[1])]) #dx, dy
        print('done')
        np.savetxt(displacement_file, displacements)
        print('saved: '+displacement_file)

    return displacements

#regの何番目がターゲット天体かどうかを返す
def get_target(circles):
    index = []
    for i in range(len(circles)):
        if circles[i].color == 'green':
            index.append(i)
    if len(index) > 1:
        print('Error: '+ str(len(index)) + ' objects have been selected as target')
        return -99
    elif len(index) == 0:
        print('Error: no objects has been selected as target')
        return -99
    return index[0]

#横軸にするヘッダーの日付を返す
def get_hdr_date(hdrs):
    dates = []
    for i in range(len(hdrs)):
        r = re.match(r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2}.\d+)',hdrs[i]['DATE-OBS'])
        dates.append(Date(int(r.group(1)),int(r.group(2)),int(r.group(3)),int(r.group(4)),int(r.group(5)),float(r.group(6))))

    return dates


"""===============================photometry=============================="""
"""===================aperture photometry====================="""
#aperture測光を行う
def do_aperture_photometry(images, displacements, circles, target, r_pix, half_aperture, half_aperture_fwhm, local_bkg = False):
    """================まず一枚目の画像からFWHMを求める==============="""
    #重心の決定
    positions, r_max = get_position_r(circles, images[0], displacements[0], r_pix) #positionと重心決定に使った半径の最大値を返す
    #FWHM
    FWHM = get_FWHMs(images[0], positions, half_aperture_fwhm, r_max, target, show_text = True)

    #apertureの決定
    if not half_aperture:
        half_aperture = FWHM/2 * 3 #FWHMの三倍をphotometryの半径にする

    #photometry
    photometries = []
    for i in tqdm.tqdm(range(len(images))):
        positions, r_max = get_position_r(circles, images[i], displacements[i], r_pix)
        photometries_temp = get_aperture_photometry(positions, images[i], half_aperture, target, local_bkg)
        photometries.append(photometries_temp)
    return photometries

#FWHMを計算 いくつかの星のうちの最大値を返す(どれか一つに固定しないと意味がないため)
def get_FWHMs(image, positions, half_aperture_fwhm, r_max, target, show_text = False):
    #設定していなければ
    if not half_aperture_fwhm:
        half_aperture_fwhm = r_max

    #アパーチャーを決めて測光
    aperture = CircularAperture(positions, r=half_aperture_fwhm) #rは半径で指定 !!
    FWHMs = ApertureStats(image, aperture).fwhm.value
    FWHM_sorted = datas_sort(FWHMs, target)

    if show_text:
        print(return_message_in_order('FWHM', FWHM_sorted, 'pix'))

    return np.max(FWHMs)

#photometryに使うpositionを返す
def get_position_r(circles, image, displacement, r_pix):
    positions = []
    rs = []
    for i in range(len(circles)):
        x_g, y_g, r = calc_centroid(image,displacement,circles[i], r_pix = r_pix)
        positions.append((x_g, y_g))
        rs.append(2*r)  #FWHMの推定に使う半径は少し大きめにとる

    return positions, np.max(np.array(rs))

#targetをindex 0として出力
def datas_sort(datas, target):
    temp = [datas[target]]
    for i in range(len(datas)):
        if i != target:
            temp.append(datas[i])
    return temp

#メッセージを返す
def return_message_in_order(message_top, data_sorted, message_bottom):
    message = message_top +': target ' + str(data_sorted[0])+', reference(s)'
    for i in range(len(data_sorted)-1):
        message+= ' ' + str(data_sorted[i+1])
    message += ' ' + message_bottom
    return message

#実際にphotometryを行う
#aperture 周辺の背景情報を取り除くこともできる
def get_aperture_photometry(positions, image, half_aperture, target, local_bkg):
    aperture = CircularAperture(positions, r=half_aperture) #rは半径で指定 !!
    aper_stats = ApertureStats(image, aperture, sigma_clip=None)
    if local_bkg: #周辺をどうするか
        annulus_aperture = CircularAnnulus(positions, r_in=half_aperture*1.5, r_out=half_aperture*2) #apertureの設定
        sigclip = SigmaClip(sigma=3.0, maxiters=10)
        bkg_stats = ApertureStats(image, annulus_aperture, sigma_clip=sigclip)#annulusの統計量 シグマクリップで
        total_bkg = bkg_stats.median * aper_stats.sum_aper_area.value #bkgの総量を計算した
    #測光
    photometries = aper_stats.sum
    if local_bkg:
        photometries = photometries - total_bkg
    return datas_sort(photometries, target)
