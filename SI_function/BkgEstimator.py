#backgroundを推定し減算するコード
import numpy as np
from photutils.segmentation import make_source_mask
from astropy.stats import sigma_clipped_stats
from SI_function import normal_func
from SI_function import astro_image
import warnings
import matplotlib.pyplot as plt
import re

"""===============マスク作成用関数==============="""
#bkg_estimateのためのマスクを作る
def make_mask(image_input, nsigma, value = False, nsigma_u = False, regfile = False, show_image = False, showtext = True):
    mask = make_source_mask(image_input, nsigma=nsigma, npixels=5, dilate_size=11) #photutils.segmentationを使いmask作成

    """========mask option======================================"""
    if value:
        mask = mask_outof_range(image_input, mask, value = value)
    if nsigma_u:
        mask = mask_under_value(image_input, mask, nsigma_u)
    if regfile:
        mask = mask_regfile_only(mask, regfile)
    """========================================================"""

    mean, median, std = sigma_clipped_stats(image_input, sigma=3, mask=mask)

    if showtext:
        print('mean = '+str(mean)+', median =  '+ str(median)+', sigma = ' +str(std))

    if show_image:
        fig=plt.figure(figsize=(10,10),dpi = 300)
        fig.patch.set_facecolor('white')
        plt.imshow(mask, cmap='Greys_r', interpolation='nearest')

    return mask, mean, median, std

#sigmaクリッピングで外せない特定の値を持つピクセルをマスクする 位置ずれの部分をマスクしたいから
def mask_outof_range(data, mask, value = 0.0):
    x = np.arange(0,len(data[0]),1)
    y = np.arange(0,len(data),1)
    XX, YY = np.meshgrid(x, y)

    c_range = np.where(abs(data-value)<1e-8)
    XX_con = XX[c_range].reshape(-1,1)
    YY_con = YY[c_range].reshape(-1,1)

    for i in range(len(XX_con)):
        mask[YY_con[i],XX_con[i]] = True

    return mask

#値の下側もマスクするとき
def mask_under_value(data, mask, nsigma_u):
    x = np.arange(0,len(data[0]),1)
    y = np.arange(0,len(data),1)
    XX, YY = np.meshgrid(x, y)

    mean, median, std = sigma_clipped_stats(data, sigma=3, mask=mask)

    c_range = np.where(data < mean-nsigma_u*std)
    XX_con = XX[c_range].reshape(-1,1)
    YY_con = YY[c_range].reshape(-1,1)

    for i in range(len(XX_con)):
        mask[YY_con[i],XX_con[i]] = True

    return mask

#region file で指定した部分のみ使うとき　ひとまず長方形のみ対応
def mask_regfile_only(mask, regfile):
    f = open(regfile, 'r')
    box = []
    for fline in f:
        r = re.match(r'box\((\d+.\d+),(\d+.\d+),(\d+.\d+),(\d+.\d+),\d+\)',fline)
        if r:
            box.append([float(r.group(1)),float(r.group(2)),float(r.group(3)),float(r.group(4))]) #x座標, y座標, x幅, y幅
    temp_data = np.zeros((len(mask),len(mask[0])))   #マスク作成のための配列を作る.

    for k in range(len(box)):
        for i in range(int(box[k][3])): #y座標
            for j in range(int(box[k][2])): # x座標
                temp_data[i+int(box[k][1])-int(box[k][3]/2)][j+int(box[k][0])-int(box[k][2]/2)] = 1
    #meshgrid
    x = np.arange(0,len(mask[0]),1)
    y = np.arange(0,len(mask),1)
    XX, YY = np.meshgrid(x, y)

    c_range = np.where(temp_data == 0 )
    XX_con = XX[c_range].reshape(-1,1)
    YY_con = YY[c_range].reshape(-1,1)

    for i in range(len(XX_con)):
        mask[YY_con[i],XX_con[i]] = True

    return mask

"""=================3D planeでfit================="""""
#データをマスクして配列化し、3Dfit関数に渡す
def masked_data_for_3D_plane(data, mask):
    #make grid
    x = np.arange(0,len(data[0]),1)
    y = np.arange(0,len(data),1)
    XX, YY = np.meshgrid(x, y)

    XX_list = XX.reshape(-1,1)
    YY_list = YY.reshape(-1,1)
    data_list = data.reshape(-1,1)

    mask_reverse_list = np.logical_not(mask.reshape(-1,1))
    c_mask =np.where(mask_reverse_list)

    X = XX_list[c_mask]
    Y = YY_list[c_mask]
    Z = data_list[c_mask]

    XYZ=np.stack([X,Y,Z],1)
    return XYZ

#fitした結果からbkg画像を作成する
def create_linear_bkg(data, a,b,c,d):
    x = np.arange(0,len(data[0]),1)
    y = np.arange(0,len(data),1)
    XX, YY = np.meshgrid(x, y)
    bkg = (-a*XX-b*YY-d)/c

    return bkg

#3D 平面でfit bkgとそれを引いたデータを返す
def sky_fitting(image_input, mask, nsigma, show_image = True, showtext = True):
    warnings.simplefilter('ignore')
    XYZ = masked_data_for_3D_plane(image_input, mask)
    a, b, c, d = normal_func.fitPlane3D(XYZ)
    bkg = create_linear_bkg(image_input, a,b,c,d)

    if show_image:
        fig=plt.figure(figsize=(3,3),dpi = 300)
        fig.patch.set_facecolor('white')
        plt.imshow(bkg, cmap='Greys_r')

    if showtext:
        mask2 = make_source_mask(image_input-bkg, nsigma=nsigma, npixels=5, dilate_size=11)
        mean, median, std = sigma_clipped_stats(image_input-bkg, sigma=3.0, mask=mask2)
        print('mean = '+str(mean)+', median =  '+ str(median)+', sigma = ' +str(std)) #ちゃんと引けてるかを確認するため

    return image_input - bkg, [a,b,c,d]

#全部まとめて実行
def sky_subtract(fname, output_name, nsigma, value, N_comp, EXPTIME, regfile = False):
    #file open
    R, G, B, header = astro_image.open_image(fname)

    #--------------R-------------
    #make mask
    r_mask, r_mean, r_median, r_sigma = make_mask(R, nsigma, value = value, regfile = regfile,showtext = False)
    #sky fitting
    R_bkg, R_parameter = sky_fitting(R, r_mask, nsigma, show_image = False, showtext = False)
    #calc_subtracted values
    r_mask, r_mean, r_median, r_sigma = make_mask(R_bkg, nsigma, value = value,showtext = False)

    #------------G-----------------
    #make mask
    g_mask, g_mean, g_median, g_sigma = make_mask(G, nsigma, value = value, regfile = regfile,showtext = False)
    #sky fitting
    G_bkg, G_parameter = sky_fitting( G, g_mask,nsigma, show_image =  False, showtext = False)
    #calc_subtracted values
    g_mask, g_mean, g_median, g_sigma = make_mask(G_bkg, nsigma, value = value,showtext = False)
    #----------------B------------------
    #make mask
    b_mask, b_mean, b_median, b_sigma = make_mask(B, nsigma, value = value, regfile = regfile,showtext = False)
    #sky fittingb
    B_bkg, B_parameter = sky_fitting(B, b_mask, nsigma, show_image = False, showtext = False)
    #calc_subtracted values
    b_mask, b_mean, b_median, b_sigma = make_mask(B_bkg, nsigma, value = value,showtext = False)

    #データキューブに変換
    RGB_bkg = np.stack([R_bkg,G_bkg,B_bkg])

    #headerに追加する項目を設定 適当に編集可
    add_header = ['N_comp',
                  'EXPTIME_INT',
                  'R_sky_mean','G_sky_mean', 'B_sky_mean',
                  'R_sky_sigma', 'G_sky_sigma', 'B_sky_sigma',
                  'R_bkg_a','R_bkg_b','R_bkg_c','R_bkg_d',
                  'G_bkg_a','G_bkg_b','G_bkg_c','G_bkg_d',
                  'B_bkg_a','B_bkg_b','B_bkg_c','B_bkg_d']
    add_header_value = [N_comp,
                        EXPTIME,
                        r_mean, g_mean, b_mean,
                        r_sigma, g_sigma, b_sigma,
                        R_parameter[0],R_parameter[1],R_parameter[2],R_parameter[3],
                        G_parameter[0],G_parameter[1],G_parameter[2],G_parameter[3],
                        B_parameter[0],B_parameter[1],B_parameter[2],B_parameter[3]]
    add_header_comment = ['number of composit',
                          'int second of exptime',
                          'r backgroud mean','g backgroud mean','b backgroud mean',
                          'r background std','g background std','b background std',
                          'r background parameter a','r background parameter b','r background parameter c','r background parameter d',
                          'g background parameter a','g background parameter b','g background parameter c','g background parameter d',
                          'b background parameter a','b background parameter b','b background parameter c','b background parameter d',]

    astro_image.output_fits(RGB_bkg, output_name, header, add_header, add_header_value, add_header_comment)

#パラメータを個別に設定したかどうか判定
def return_params(nsigmas, values, N_comps, EXPTIMEs, i):
    if len(nsigmas)>1:
        nsigma = nsigmas[i]
    else:
        nsigma = nsigmas[0]

    if len(values)>1:
        value = values[i]
    else:
        value = values[0]

    if len(N_comps)>1:
        N_comp = N_comps[i]
    else:
        N_comp = N_comps[0]

    if len(EXPTIMEs)>1:
        EXPTIME = EXPTIMEs[i]
    else:
        EXPTIME = EXPTIMEs[0]

    return nsigma, value, N_comp, EXPTIME
