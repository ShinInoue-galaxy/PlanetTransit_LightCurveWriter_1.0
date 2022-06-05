import tqdm
from SI_function import ImageReduction as IR
import numpy as np
from SI_function import astro_image

"""============parameters=============="""


dark_list = '../WASP-43/Dark/dark_50.list'  #darkファイルがある場所に置く 必ずdark.listの形式
output_file = 'dark_50_clip.FIT'  #ファイル名だけ
sigma_clip = True # σクリッピングをするかどうか
n_sigma = 1. #何σでクリップするか


"""===================================="""
PATH, output_name = IR.set_PATH_name(dark_list, output_file, type = 'dark')
#配列でまとめてオープン
images, hdrs = IR.open_fits_multi(dark_list, PATH)

image_composit = []
for i in range(len(images[0])):
    image_composit.append(IR.sigma_clip_average(i, images, n_sigma))

astro_image.output_fits(np.array(image_composit), output_name, header = hdrs[0])
