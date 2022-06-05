#ライトカーブを求めるのに使う関数
#module import
import re
import tqdm
import os
import numpy as np
import image_registration as im_regi # image_registrationパッケージを読み込む
from SI_function import normal_func

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

    return x_g, y_g


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

def get_hdr_date(hdrs):
    dates = []
    for i in range(len(hdrs)):
        r = re.match(r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2}.\d+)',hdrs[i]['DATE-OBS'])
        dates.append(Date(int(r.group(1)),int(r.group(2)),int(r.group(3)),int(r.group(4)),int(r.group(5)),float(r.group(6))))

    return dates
