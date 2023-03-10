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
import math
import torch
import torch.nn as nn
import warnings

from compressai.models.priors import CompressionModel, GaussianConditional
from compressai.ops import ste_round
from compressai.layers import GDN, MaskedConv2d
from compressai.models.utils import conv, deconv, update_registered_buffers

from ptflops import get_model_complexity_info

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
eps = 1e-9
def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class Quantizer():
    def quantize(self, inputs, quantize_type="noise"):
        if quantize_type == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs
        elif quantize_type == "ste":
            return torch.round(inputs) - inputs.detach() + inputs
        else:
            return torch.round(inputs)

class ScaleHyperprior(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.
    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=192,  M=320, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

        self.quantizer = Quantizer()
        self.lmbda = [0.0018, 0.0035, 0.0067, 0.0130, 0.025, 0.0483, 0.0932, 0.18]
        self.Gain = torch.nn.Parameter(torch.tensor(
            [1.0000, 1.3944, 1.9293, 2.6874, 3.7268, 5.1801, 7.1957, 10.0000]), requires_grad=True)
        self.levels = len(self.lmbda)  # 8

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x, noise=False, stage=3, s=1):
        if stage > 1:
            if s != 0:
                scale = torch.max(self.Gain[s], torch.tensor(1e-4)) + eps
            else:
                s = 0
                scale = self.Gain[s].detach()
        else:
            scale = self.Gain[0].detach()

        rescale = 1.0 / scale.clone().detach()

        if noise:
            y = self.g_a(x)
            z = self.h_a(torch.abs(y))
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            scales_hat = self.h_s(z_hat)
            y_hat = self.gaussian_conditional.quantize(y * scale,
                                                       "noise" if self.training else "dequantize") * rescale
            _, y_likelihoods = self.gaussian_conditional(y * scale, scales_hat * scale)
            x_hat = self.g_s(y_hat)
        else:
            y = self.g_a(x)
            z = self.h_a(torch.abs(y))
            z_hat, z_likelihoods = self.entropy_bottleneck(z)

            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = ste_round(z_tmp) + z_offset

            scales_hat = self.h_s(z_hat)
            y_hat = self.quantizer.quantize(y * scale, "ste") * rescale
            _, y_likelihoods = self.gaussian_conditional(y * scale, scales_hat * scale)
            x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x, s, inputscale=0):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )
        if inputscale != 0:
            scale = inputscale
        else:
            assert s in range(0, self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
            scale = torch.abs(self.Gain[s])

        rescale = torch.tensor(1.0) / scale
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat * scale)
        y_strings = self.gaussian_conditional.compress(y * scale, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape, s, inputscale):
        assert isinstance(strings, list) and len(strings) == 2
        if inputscale != 0:
            scale = inputscale
        else:
            assert s in range(0, self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
            scale = torch.abs(self.Gain[s])

        rescale = torch.tensor(1.0) / scale

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat * scale)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes) * rescale
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

if __name__ == "__main__":
    model = ScaleHyperprior(N=192, M=320)
    input = torch.Tensor(1, 3, 256, 256)
    print(model)
    out = model(input)
    print(out["x_hat"].shape)
    # flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True,
    #                                           flops_units='Mac', param_units='')
    # print('flops: ', flops, 'params: ', params)
