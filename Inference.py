# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

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

# This file may have been modified by Bytedance Inc. (“Bytedance Modifications”). All Bytedance Modifications are Copyright 2022 Bytedance Inc.


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

import argparse
import json
import sys
import time
import csv

from collections import defaultdict
from typing import List
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms
import torchvision
import compressai
from compressai.zoo import load_state_dict
import torch
import os
import math
import torch.nn as nn
from Cheng2020Attention import Cheng2020Attention


torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

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

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.nn.functional.mse_loss(a, b).item()
    return -10 * math.log10(mse)

def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)

@torch.no_grad()
def inference(model, x, f, outputpath, patch, s, factor, factormode):
    x = x.unsqueeze(0)
    imgpath = f.split('/')
    imgpath[-2] = outputpath
    imgPath = '/'.join(imgpath)
    csvfile = '/'.join(imgpath[:-1]) + '/'+outputpath+'_result.csv'
    print('decoding img: {}'.format(f))
    h, w = x.size(2), x.size(3)
    p = patch  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = 0
    padding_right = new_w - w - padding_left
    padding_top = 0
    padding_bottom = new_h - h - padding_top
    pad = nn.ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 0)
    x_padded = pad(x)

    _, _, height, width = x_padded.size()
   
    start = time.time()
    out_enc = model.compress(x_padded, s, factor)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"], s, factor)
    dec_time = time.time() - start

    
    out_dec["x_hat"] = torch.nn.functional.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    z_bpp = len(out_enc["strings"][1][0])* 8.0 / num_pixels
    y_bpp = bpp - z_bpp

    torchvision.utils.save_image(out_dec["x_hat"], imgPath, nrow=1)
    PSNR = psnr(x, out_dec["x_hat"])
    with open(csvfile, 'a+') as f:
        row = [imgpath[-1], bpp * num_pixels, num_pixels, bpp, y_bpp, z_bpp,
               torch.nn.functional.mse_loss(x, out_dec["x_hat"]).item() * 255 ** 2, psnr(x, out_dec["x_hat"]),
               ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(), enc_time, dec_time]
        write = csv.writer(f)
        write.writerow(row)
    print('bpp:{}, PSNR: {}, encoding time: {}, decoding time: {}'.format(bpp, PSNR, enc_time, dec_time))
    return {
        "psnr": PSNR,
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }

@torch.no_grad()
def inference_entropy_estimation(model, x, f, outputpath, patch, s):
    x = x.unsqueeze(0)
    imgpath = f.split('/')
    imgpath[-2] = outputpath
    imgPath = '/'.join(imgpath)
    csvfile = '/'.join(imgpath[:-1]) + '/'+outputpath+'_result.csv'
    print('decoding img: {}'.format(f))
    h, w = x.size(2), x.size(3)
    p = patch  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = torch.nn.functional.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    _, _, height, width = x_padded.size()

    start = time.time()
    out_net = model.forward(x_padded, noisequant=False, training_stage=3, s=s)

    elapsed_time = time.time() - start
    out_net["x_hat"] = torch.nn.functional.pad(
        out_net["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )
    y_bpp = (torch.log(out_net["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels))
    z_bpp = (torch.log(out_net["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels))

    torchvision.utils.save_image(out_net["x_hat"], imgPath, nrow=1)
    PSNR = psnr(x, out_net["x_hat"])
    with open(csvfile, 'a+') as f:
        row = [imgpath[-1], bpp.item() * num_pixels, num_pixels, bpp.item(), y_bpp.item(), z_bpp.item(),
               torch.nn.functional.mse_loss(x, out_net["x_hat"]).item() * 255 ** 2, PSNR,
               ms_ssim(x, out_net["x_hat"], data_range=1.0).item(), elapsed_time / 2.0, elapsed_time / 2.0,]
        write = csv.writer(f)
        write.writerow(row)
    return {
        "psnr": PSNR,
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def eval_model(model, filepaths, entropy_estimation=False, half=False, outputpath='Recon', patch=64, s=2, factor=0, factormode=False):
    print("variable rate s:{}".format(s))
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    imgdir = filepaths[0].split('/')
    imgdir[-2] = outputpath
    imgDir = '/'.join(imgdir[:-1])
    if not os.path.isdir(imgDir):
        os.makedirs(imgDir)
    csvfile = imgDir + '/'+outputpath+'_result.csv'
    if os.path.isfile(csvfile):
        os.remove(csvfile)
    with open(csvfile, 'w') as f:
        row = ['name', 'bits', 'pixels', 'bpp', 'y_bpp', 'z_bpp', 'mse', 'psnr(dB)', 'ms-ssim', 'enc_time(s)', 'dec_time(s)',]
        write = csv.writer(f)
        write.writerow(row)
    for f in filepaths:
        x = read_image(f).to(device)
        if not entropy_estimation:
            if half:
                model = model.half()
                x = x.half()
            rv = inference(model, x, f, outputpath, patch, s, factor, factormode)
        else:
            assert not factormode, f"entropy estimation not support factormode"
            rv = inference_entropy_estimation(model, x, f, outputpath, patch, s)
        for k, v in rv.items():
            metrics[k] += v
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics

def setup_args():
    parser = argparse.ArgumentParser(
        add_help=False,
    )

    # Common options.
    parser.add_argument("--dataset", type=str, help="dataset path")
    parser.add_argument(
        "--output_path",
        help="result output path",
    )
    parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="enable CUDA",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parser.add_argument(
        "-p",
        "--path",
        dest="paths",
        type=str,
        required=True,
        help="checkpoint path",
    )
    parser.add_argument(
        "--factormode",
        type=int,
        default=0,
        help="weather to use factor mode",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=1.5,
        help="choose the value of factor",
    )
    parser.add_argument(
        "--patch",
        type=int,
        default=64,
        help="padding patch size (default: %(default)s)",
    )
    parser.add_argument(
        "--s",
        type=int,
        default=2,
        help="select the scale factor",
    )
    return parser


def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)

    filepaths = collect_images(args.dataset)
    filepaths = sorted(filepaths)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        sys.exit(1)

    compressai.set_entropy_coder(args.entropy_coder)

    model = Cheng2020Attention()
    checkpoint = torch.load(args.paths, map_location="cpu")
    if "state_dict" in checkpoint:
        ckpt = checkpoint["state_dict"]
    else:
        ckpt = checkpoint
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt.items() if
                       k in model_dict.keys() and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model_cls = Cheng2020Attention()
    model = model_cls.from_state_dict(model_dict).eval()
    model.update(force=True)

    results = defaultdict(list)
    if args.cuda and torch.cuda.is_available():
        model = model.to("cuda")

    if args.factormode:
        if 0.5<=args.factor<=12:
            metrics = eval_model(model, filepaths, args.entropy_estimation, args.half,
                                 args.output_path + '_factor_' + str(args.factor),
                                 args.patch, s=2, factor=args.factor, factormode=args.factormode)
            for k, v in metrics.items():
                results[k].append(v)
        else:
            for factor in [0.5, 0.7, 0.9, 1.1, 1.25, 1.45, 1.7, 2.0, 2.4, 2.8, 3.3, 3.8, 4.0,
            4.6, 5.4, 5.8, 6.5, 6.8, 7.5, 7.9, 8.3, 9.1, 9.4, 9.7, 10.5, 11, 12]:
                metrics = eval_model(model, filepaths, args.entropy_estimation, args.half,
                                     args.output_path + '_factor_' + str(factor),
                                     args.patch, s=2, factor=factor, factormode=args.factormode)
                for k, v in metrics.items():
                    results[k].append(v)
    elif 0<=args.s<=model.levels:
            metrics = eval_model(model, filepaths, args.entropy_estimation, args.half,
                                 args.output_path + '_s_' + str(args.s),
                                 args.patch, args.s, factor=0, factormode=0)
            for k, v in metrics.items():
                results[k].append(v)
    else:
        for s in range(model.levels):
            metrics = eval_model(model, filepaths, args.entropy_estimation, args.half, args.output_path + '_s_' + str(s),
                                args.patch, s, factor=0, factormode=0)
            for k, v in metrics.items():
                results[k].append(v)


    description = (
        "entropy estimation" if args.entropy_estimation else args.entropy_coder
    )
    output = {
        "description": f"Inference ({description})",
        "results": results,
    }
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main(sys.argv[1:])
