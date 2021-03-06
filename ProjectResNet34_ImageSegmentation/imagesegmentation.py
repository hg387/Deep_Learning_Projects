# -*- coding: utf-8 -*-
"""ImageSegmentation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ysWLuYaOAT01ei6xLuLF831HLFSY4w0V
"""

!curl -s https://course.fast.ai/setup/colab | bash
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
root_dir = "/content/gdrive/My Drive/"
base_dir = root_dir + 'fastai/'

from fastai import *
from fastai.vision import *

path = untar_data(URLs.CAMVID)
path_lbl = path/'labels'
path_img = path/'images'

fnames = get_image_files(path_img)
lbl_names = get_image_files(path_lbl)

get_y_fn = lambda x: path_lbl/f'{x.stem}_P{x.suffix}'

img_f = fnames[0]
mask = open_mask(get_y_fn(img_f))

src_size = np.array(mask.shape[1:])
codes = np.loadtxt(path/'codes.txt', dtype=str); codes
size = src_size//2
bs=4

src = (SegmentationItemList.from_folder(path_img)
       .split_by_fname_file('../valid.txt')
       .label_from_func(get_y_fn, classes=codes))

data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Void']

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

metrics=acc_camvid
# metrics=accuracy

learn = unet_learner(data, models.resnet34, metrics=metrics)

lr_find(learn)
learn.recorder.plot()

lr=1e-3
learn.fit_one_cycle(10, slice(lr))

learn.unfreeze()
lr_find(learn)
learn.recorder.plot()

lrs = slice(1e-4,lr/5)
learn.fit_one_cycle(10, lrs)

learn.save('step1')

learn.recorder.plot_lr()

learn.recorder.plot_losses()

size = src_size
bs=1
data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))
learn = unet_learner(data, models.resnet34, metrics=metrics)
learn.load('step1');
lr_find(learn)
learn.recorder.plot()

lr=1e-4
learn.fit_one_cycle(10, slice(lr))

learn.recorder.plot_losses()

learn.save('step2')

learn.load('step2')

learn.unfreeze()
lr_find(learn)
learn.recorder.plot()

learn.show_results()

lrs = slice(1e-6,1e-5)
learn.fit_one_cycle(5, lrs, wd=1e-3)

learn.recorder.plot_losses()