#マスターフラットを作る。どうにもσクリッピングが必要そうなので並列化して計算。
#module import
import multiprocessing
from multiprocessing import Pool
import tqdm
from SI_function import ImageReduction as IR
import numpy as np
from SI_function import astro_image

"""============parameters=============="""
if __name__ == "__main__":

    flat_list = 'PATH/flat.list'  #darkファイルがある場所に置く 必ずdark.listの形式
    dark_file = 'PATH/out/dark.FIT' #フルパス
    output_file = 'PATH/flat.FIT'  #ファイル名だけ
    sigma_clip = True # σクリッピングをするかどうか
    n_sigma = 1. #何σでクリップするか
"""===================================="""


"""===================main================="""
proc = multiprocessing.cpu_count()

#--------------- set for parallel----------
if __name__ == "__main__":
    print('use ' +str(proc) + ' cpus')
    print()

    if sigma_clip: #シグマクリップをする場合 並列で計算する
        p = Pool(proc)
        #pathと出力ファイル名の決定
        PATH, output_name = IR.set_PATH_name(flat_list, output_file, type = 'flat')
        #配列でまとめてオープン
        images, hdrs = IR.open_fits_multi(flat_list, PATH)

        # ダーク減算
        images = IR.image_subtract_dark_multi(images, dark_file)

        #sigma_clipで計算。
        """==============並列計算==========="""
        x = np.arange(0,len(images[0]),1)
        imagess = [images]
        n_sigmas = [n_sigma]

        print('sigma clipping')
        args = [(i, images, n_sigma) for i in x for images in imagess for n_sigma in n_sigmas]
        image_composit = p.map(IR.sigma_clip_average_wrapper, tqdm.tqdm(args))

        astro_image.output_fits(np.array(image_composit), output_name, header = hdrs[0])

    else: #普通に重ね合わせて終わり。
        IR.make_flat(flat_list, dark_file, output_file)
