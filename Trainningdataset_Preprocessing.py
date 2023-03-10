
# Copyright 2023 Bytedance Inc.
# All rights reserved.
# Licensed under the BSD 3-Clause Clear License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://choosealicense.com/licenses/bsd-3-clause-clear/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

###preprocessing the maximum size of 8000 of ILSVRC training dataset and CLIC trainning with reshape and add normal noise
import os.path
from pathlib import Path

from PIL import Image
import cv2
import numpy as np
import os
from typing import List

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]

def preprocessing(imgdir, savedir):
    """

    :param imgdir: input ILSVRC largest 8000 images
    :param savedir: the proprecessed image save dir
    :return:
    Add noise
    "Checkerboard Context Model for Efficient Learned Image Compression"
    Following previous works [5, 6], we add random uniform noise to each of them and then downsample all the images.
    Reshaping
    "VARIATIONAL IMAGE COMPRESSION WITH A SCALE HYPERPRIOR"
    The models were trained on a body of color JPEG images with heights/widths between 3000 and
    5000 pixels, comprising approximately 1 million images scraped from the world wide web. Images
    with excessive saturation were screened out to reduce the number of non-photographic images.
    To reduce existing compression artifacts, the images were further downsampled by a randomized
    factor, such that the minimum of their height and width equaled between 640 and 1200 pixels. Then,
    randomly placed 256x256 pixel crops of these downsampled images were extracted.
    """
    imgdir = Path(imgdir)
    savedir = Path(savedir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    if not imgdir.is_dir():
        raise RuntimeError(f'Invalid directory "{imgdir}"')

    img_paths = collect_images(imgdir)
    for imgpath in img_paths:
        img = cv2.imread(imgpath)
        img = np.array((img)).astype(('float64'))
        height, width, channel = img.shape
        ### adding unifor noise
        noise = np.random.uniform(0, 1, (height, width, channel)).astype('float32')
        img += noise
        img = img.astype('uint8')
        if min(width, height)>512:
            img = cv2.resize(img, dsize=((int(width//2), int(height//2))), interpolation=cv2.INTER_CUBIC)
        name = os.path.splitext(os.path.basename(imgpath))[0]
        cv2.imwrite(os.path.join(savedir, name + '.png'), img)

def select_n_images(imgdir, savedir, n):
    """

    :param imgdir: input image dir
    :param savedir: seleted image savingdir
    :param n: the largest n images in the imgdir
    :return:
    """
    import bisect
    import shutil
    import imagesize
    imgdir = Path(imgdir)
    savedir = Path(savedir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    if not imgdir.is_dir():
        raise RuntimeError(f'Invalid directory "{imgdir}"')

    img_paths = collect_images(imgdir)
    sizepath = []
    namepath = []
    for imgpath in img_paths:
        width, height = imagesize.get(imgpath)
        size = width*height
        loc = bisect.bisect_left(sizepath, size)
        sizepath.insert(loc, size)
        namepath.insert(loc, imgpath)
        if len(sizepath)>n:
            sizepath.pop(0)
            namepath.pop(0)
    for path in namepath:
        imgname = os.path.basename(path)
        shutil.copyfile(path, os.path.join(savedir, imgname))


if __name__ == '__main__':
    inputimagedir = './dataset/CLIC2021_train' ## original image dataset
    tmpdir = './dataset/tmp' ## temporary image folder
    savedir = './CLIC2021_train_dowmsample' ## preprocessed image folder
    select_n_images(inputimagedir, tmpdir, 8000) ## select 8000 images from ImageNet training dataset, and all image from CLIC training
    preprocessing(tmpdir, savedir)
