#天体画像に対して使う操作をまとめた関数。
#module import
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as fits

#モノクロデータのみに対応
#fitsを保存する関数 data outputname + header(optional)
def output_fits(data, output_name, header=False, add_header=False, add_header_value=False, add_header_comment = False):
    #ヘッダの設定
    if header:
        hdu_bkg = fits.PrimaryHDU(data = data, header = header)
    else:
        hdu_bkg = fits.PrimaryHDU(data = data)

    #追加のheaderがある場合にヘッダーカードを追加
    if add_header:
        if add_header_value:
            if add_header_comment:
                for i in range(len(add_header)):
                    hdu_bkg.header.set(add_header[i], add_header_value[i], add_header_comment[i])
            else:
                for i in range(len(add_header)):
                    hdu_bkg.header.set(add_header[i], add_header_value[i])

    #hduを作り出力
    hdulist = fits.HDUList([hdu_bkg])
    hdulist.writeto(output_name,overwrite=True)
    print('save '+output_name )

#----------------prow-------------------------
#iraf prowに相当
def prow(data, row,xmin = 0,xmax =-1, log = False):
    #指定したrowのデータを取り出し
    data_row = data[row,:]

    #まず二次元の画像上で選択した部分を表示
    fig=plt.figure(figsize=(12,8),dpi = 300)
    fig.patch.set_facecolor('white')
    a = 1000
    data_log = image_log(data)
    plt.imshow(data_log,vmin = 0, vmax = 1, cmap = 'gray')
    plt.axhline(row,0,1,color = 'red',linewidth = 1)

    #row上のデータを表示
    fig=plt.figure(figsize=(4,3),dpi = 300)
    fig.patch.set_facecolor('white')
    if log:
        data_input = np.log(abs(data_row)+1e-8)
    else:
        data_input = data_row
    plt.plot(data_input[xmin:xmax], linewidth = 1)


#----------------pcol-------------------------
#iraf pcolに相当
def pcol(data, col,xmin = 0,xmax =-1, log = False):
    #指定したcolのデータを取り出し
    data_col = data[:,col]

    #まず二次元の画像上で選択した部分を表示
    fig=plt.figure(figsize=(12,8),dpi = 300)
    fig.patch.set_facecolor('white')
    a = 1000
    data_log = image_log(data)
    plt.imshow(data_log,vmin = 0, vmax = 1, cmap = 'gray')
    plt.axvline(col,0,1,color = 'red',linewidth = 1)

    #col上のデータを表示
    fig=plt.figure(figsize=(4,3),dpi = 300)
    fig.patch.set_facecolor('white')
    if log:
        data_input = np.log(abs(data_col)+1e-8)
    else:
        data_input = data_col
    plt.plot(data_input[xmin:xmax], linewidth = 1)

#----------------log as in ds9-----------------
#ds9　形式のlog
def image_log(data, a = 1000):
    return np.log(a*(data-np.min(data.reshape(-1,1)))/(np.max(data.reshape(-1,1))-np.min(data.reshape(-1,1)))+1)/np.log(a)
