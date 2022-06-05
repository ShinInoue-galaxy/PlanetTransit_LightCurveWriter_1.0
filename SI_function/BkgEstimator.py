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
#bkg_estimationのためのマスクを作る
def make_mask(image_input, nsigma, value = False, nsigma_u = False, regfile = False, show_image = False, showtext = True):
    #まずはマスクを作成
    #photutils.segmentation.make_source_maskであるσ以上の連続分布をマスク　(星・星雲をマスク)
    mask = make_source_mask(image_input, nsigma=nsigma, npixels=5, dilate_size=11) #photutils.segmentationを使いmask作成

    #マスク領域の追加
    """========mask option======================================"""
    if value: #ある値のデータは全てマスクする　(コンポジット範囲外など)
        mask = mask_outof_range(image_input, mask, value = value)
    if nsigma_u: #下側のみのσクリッピング #未動作確認
        mask = mask_under_value(image_input, mask, nsigma_u)
    if regfile: #ds9.reg形式で出力した領域内のみを近似に使う　現状boxのみ対応
        mask = mask_regfile_only(mask, regfile)
    """========================================================"""

    #maskされてない部分（スカイ部分）の統計
    mean, median, std = sigma_clipped_stats(image_input, sigma=3, mask=mask)

    #値のひょうじ
    if showtext:
        print('mean = '+str(mean)+', median =  '+ str(median)+', sigma = ' +str(std))

    #マスクの表示
    if show_image:
        fig=plt.figure(figsize=(10,10),dpi = 300)
        fig.patch.set_facecolor('white')
        plt.imshow(mask, cmap='Greys_r', interpolation='nearest')

    return mask, mean, median, std

#sigmaクリッピングで外せない特定の値を持つピクセルをマスクする コンポジット範囲外をマスク
def mask_outof_range(data, mask, value = 0.0):
    #meshgridで配列作成
    x = np.arange(0,len(data[0]),1)
    y = np.arange(0,len(data),1)
    XX, YY = np.meshgrid(x, y)

    #条件部分をマスク
    c_range = np.where(abs(data-value)<1e-8)
    XX_con = XX[c_range].reshape(-1,1)
    YY_con = YY[c_range].reshape(-1,1)

    for i in range(len(XX_con)):
        mask[YY_con[i],XX_con[i]] = True

    return mask

#下側σクリッピングマスク
def mask_under_value(data, mask, nsigma_u):
    #meshgridで配列作成
    x = np.arange(0,len(data[0]),1)
    y = np.arange(0,len(data),1)
    XX, YY = np.meshgrid(x, y)

    mean, median, std = sigma_clipped_stats(data, sigma=3, mask=mask)

    #条件部分をマスク
    c_range = np.where(data < mean-nsigma_u*std)
    XX_con = XX[c_range].reshape(-1,1)
    YY_con = YY[c_range].reshape(-1,1)

    for i in range(len(XX_con)):
        mask[YY_con[i],XX_con[i]] = True

    return mask

#region file で指定した部分のみ使うとき　ひとまず長方形のみ対応
def mask_regfile_only(mask, regfile):
    #region ファイルの読み込みとbox抽出
    f = open(regfile, 'r')
    box = []
    for fline in f:
        r = re.match(r'box\((\d+.\d+),(\d+.\d+),(\d+.\d+),(\d+.\d+),\d+\)',fline)
        if r:
            box.append([float(r.group(1)),float(r.group(2)),float(r.group(3)),float(r.group(4))]) #x座標, y座標, x幅, y幅
    temp_data = np.zeros((len(mask),len(mask[0])))   #マスク作成のための配列を作る.

    #box内のみ選択
    for k in range(len(box)):
        for i in range(int(box[k][3])): #y座標
            for j in range(int(box[k][2])): # x座標
                temp_data[i+int(box[k][1])-int(box[k][3]/2)][j+int(box[k][0])-int(box[k][2]/2)] = 1

    #meshgridで配列作成
    x = np.arange(0,len(mask[0]),1)
    y = np.arange(0,len(mask),1)
    XX, YY = np.meshgrid(x, y)

    #条件部分をマスク
    c_range = np.where(temp_data == 0 )
    XX_con = XX[c_range].reshape(-1,1)
    YY_con = YY[c_range].reshape(-1,1)

    for i in range(len(XX_con)):
        mask[YY_con[i],XX_con[i]] = True

    return mask


"""=================3D planeでfit================="""""
#マスクされてない部分のみを抽出 3D plane fitに渡す
def masked_data_for_3D_plane(data, mask):
    #mesh grid
    x = np.arange(0,len(data[0]),1)
    y = np.arange(0,len(data),1)
    XX, YY = np.meshgrid(x, y)

    #一次元配列化
    XX_list = XX.reshape(-1,1)
    YY_list = YY.reshape(-1,1)
    data_list = data.reshape(-1,1)

    #条件適用
    mask_reverse_list = np.logical_not(mask.reshape(-1,1))
    c_mask =np.where(mask_reverse_list)

    #データ抽出
    X = XX_list[c_mask]
    Y = YY_list[c_mask]
    Z = data_list[c_mask]

    XYZ=np.stack([X,Y,Z],1)
    return XYZ

#fitした結果からbkgの二次元画像を作成する
def create_linear_bkg(data, a,b,c,d):
    x = np.arange(0,len(data[0]),1)
    y = np.arange(0,len(data),1)
    XX, YY = np.meshgrid(x, y)
    bkg = (-a*XX-b*YY-d)/c

    return bkg

#3D 平面でfit bkgとそれを引いたデータを返す
#作業の総まとめ関数
def sky_fitting(image_input, mask, nsigma, show_image = True, showtext = True):
    #警告が出るので無視
    warnings.simplefilter('ignore')

    #fitに使うデータのみを取得
    XYZ = masked_data_for_3D_plane(image_input, mask)
    #fitして
    a, b, c, d = normal_func.fitPlane3D(XYZ)
    #二次元画像を取得
    bkg = create_linear_bkg(image_input, a,b,c,d)

    #sky画像の表示
    if show_image:
        fig=plt.figure(figsize=(3,3),dpi = 300)
        fig.patch.set_facecolor('white')
        plt.imshow(bkg, cmap='Greys_r')

    #統計情報
    if showtext:
        mask2 = make_source_mask(image_input-bkg, nsigma=nsigma, npixels=5, dilate_size=11)
        mean, median, std = sigma_clipped_stats(image_input-bkg, sigma=3.0, mask=mask2)
        print('mean = '+str(mean)+', median =  '+ str(median)+', sigma = ' +str(std)) #ちゃんと引けてるかを確認するため

    return image_input - bkg, [a,b,c,d]
