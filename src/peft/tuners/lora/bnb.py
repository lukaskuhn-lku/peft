# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import List, Optional

import bitsandbytes as bnb
import torch

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils.other import transpose

from .layer import LoraLayer


if is_bnb_available():

    class Linear8bitLt(torch.nn.Module, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: bool = True,
            **kwargs,
        ) -> None:
            super().__init__()
            LoraLayer.__init__(self, base_layer)

            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)

        def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
            """
            Merge the active adapter weights into the base weights

            Args:
                safe_merge (`bool`, *optional*):
                    If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                    before merging the weights. This is useful if you want to check if the merge operation will produce
                    NaNs. Defaults to `False`.
                adapter_names (`List[str]`, *optional*):
                    The list of adapter names that should be merged. If None, all active adapters will be merged.
                    Defaults to `None`.
            """
            if self.merged:
                warnings.warn(
                    f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                    f"You are now additionally merging {','.join(self.active_adapters)}."
                )

            if adapter_names is None:
                adapter_names = self.active_adapters

            for active_adapter in adapter_names:
                if active_adapter not in self.lora_A.keys():
                    continue
                warnings.warn(
                    "Merge lora module to 8-bit linear may get different generations due to rounding errors."
                )
                lora_data = self.get_delta_weight(active_adapter)

                weight = self.get_base_layer().weight
                state = self.get_base_layer().state
                if state.SCB is None:
                    state.SCB = weight.SCB

                # Dequantize the result of identity matrix and int8 weight because bitsandbytes does not support int8
                # dequantization directly
                im = torch.eye(weight.data.shape[-1]).contiguous().half().to(weight.device)
                im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
                im, Sim = bnb.functional.transform(im, "col32")
                if state.CxB is None:
                    state.CxB, state.SB = bnb.functional.transform(weight.data, to_order=state.formatB)
                out32, Sout32 = bnb.functional.igemmlt(im, state.CxB, Sim, state.SB)
                output = bnb.functional.mm_dequant(out32, Sout32, SCim, state.SCB, bias=None).t()

                w_data = output.to(lora_data.dtype).to(lora_data.device) + lora_data
                if safe_merge and not torch.isfinite(w_data).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )

                self.get_base_layer().weight = bnb.nn.Int8Params(
                    w_data.to("cpu"), requires_grad=False, has_fp16_weights=weight.has_fp16_weights
                ).to(weight.device)
                state.reset_grads()
                self.merged_adapters.append(active_adapter)

        def unmerge(self):
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return

            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter not in self.lora_A.keys():
                    continue
                warnings.warn(
                    "Unmerge lora module to 8-bit linear may get different generations due to rounding errors."
                )
                lora_data = self.get_delta_weight(active_adapter)

                weight = self.get_base_layer().weight
                state = self.get_base_layer().state
                if state.SCB is None:
                    state.SCB = weight.SCB
                im = torch.eye(weight.data.shape[-1]).contiguous().half().to(weight.device)
                im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
                im, Sim = bnb.functional.transform(im, "col32")

                if state.CxB is None:
                    state.CxB, state.SB = bnb.functional.transform(weight.data, to_order=state.formatB)
                out32, Sout32 = bnb.functional.igemmlt(im, state.CxB, Sim, state.SB)
                output = bnb.functional.mm_dequant(out32, Sout32, SCim, state.SCB, bias=None).t()

                w_data = output.to(lora_data.dtype).to(lora_data.device) - lora_data
                self.get_base_layer().weight = bnb.nn.Int8Params(
                    w_data.to("cpu"), requires_grad=False, has_fp16_weights=weight.has_fp16_weights
                ).to(weight.device)
                state.reset_grads()

        def get_delta_weight(self, adapter):
            return (
                transpose(
                    self.lora_B[adapter].weight @ self.lora_A[adapter].weight,
                    False,
                )
                * self.scaling[adapter]
            )

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                result = self.base_layer(x, *args, **kwargs)
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.lora_A.keys():
                        continue
                    lora_A = self.lora_A[active_adapter]
                    lora_B = self.lora_B[active_adapter]
                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        compute_dtype = lora_A.weight.dtype
                        if x.dtype != compute_dtype:
                            x = x.to(compute_dtype)
                    output = lora_B(lora_A(dropout(x)))
                    if requires_conversion:
                        output = output.to(expected_dtype)
                    output = output * scaling
                    result += output

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "lora." + rep


if is_bnb_4bit_available():

    class Linear4bit(torch.nn.Module, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: bool = True,
            **kwargs,
        ) -> None:
            super().__init__()

            self.qa = kwargs["quantization_awareness"]
            if self.qa:
                self.block_size = 64 # BNB4bit uses 64 block size
                self.qa_pool = torch.nn.AvgPool1d(self.block_size) 
                base_layer.in_features = base_layer.in_features // self.block_size 

            LoraLayer.__init__(self, base_layer)

            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)

        def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
            """
            Merge the active adapter weights into the base weights

            Args:
                safe_merge (`bool`, *optional*):
                    If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                    before merging the weights. This is useful if you want to check if the merge operation will produce
                    NaNs. Defaults to `False`.
                adapter_names (`List[str]`, *optional*):
                    The list of adapter names that should be merged. If None, all active adapters will be merged.
                    Defaults to `None`.
            """
            if self.merged:
                warnings.warn(
                    f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                    f"You are now additionally merging {','.join(self.active_adapters)}."
                )

            if adapter_names is None:
                adapter_names = self.active_adapters

            for active_adapter in adapter_names:
                if active_adapter not in self.lora_A.keys():
                    continue
                warnings.warn(
                    "Merge lora module to 4-bit linear may get different generations due to rounding errors."
                )
                # Refer to https://gist.github.com/ChrisHayduk/1a53463331f52dca205e55982baf9930
                weight = self.get_base_layer().weight
                kwargs = weight.__dict__
                lora_data = self.get_delta_weight(active_adapter)

                if self.qa:
                    shape = self.base_layer.weight.quant_state[1]
                    # Create full size c and full size lora (dimensions of base_layer)
                    c = (127 / weight.quant_state[0]).view(shape[0], shape[1]//self.block_size).unsqueeze(2).expand(-1, -1, 64).reshape(shape[0], shape[1])
                    lora_fullsize = lora_data.view(-1).view(shape[0], shape[1]//self.block_size).unsqueeze(2).expand(-1, -1, 64).reshape(shape[0], shape[1])

                    w_dequantized = bnb.functional.dequantize_4bit(weight.data, weight.quant_state, quant_type=kwargs['quant_type'])

                    # Implementation of QLora quantization technique with QALora awareness
                    w_and_lora = (w_dequantized + lora_fullsize) * c
                    w_data = torch.round(w_and_lora)
                else:
                    w_data = bnb.functional.dequantize_4bit(weight.data, weight.quant_state) + lora_data

                if safe_merge and not torch.isfinite(w_data).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )

                if self.qa:
                    # Set base_layer weights to new data and use that in QALinear4Bit layer
                    self.base_layer.weight.data = w_data
                else:
                    self.get_base_layer().weight = bnb.nn.Params4bit(w_data.to("cpu"), requires_grad=False, **kwargs).to(
                        weight.device
                    )

                self.merged_adapters.append(active_adapter)

        def unmerge(self):
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return

            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter not in self.lora_A.keys():
                    continue
                warnings.warn(
                    "Unmerge lora module to 4-bit linear may get different generations due to rounding errors."
                )
                weight = self.get_base_layer().weight
                kwargs = weight.__dict__
                lora_data = self.get_delta_weight(active_adapter)
                w_data = bnb.functional.dequantize_4bit(weight.data, weight.quant_state) - lora_data
                self.get_base_layer().weight = bnb.nn.Params4bit(w_data.to("cpu"), requires_grad=False, **kwargs).to(
                    weight.device
                )

        def get_delta_weight(self, adapter):
            return (
                transpose(
                    self.lora_B[adapter].weight @ self.lora_A[adapter].weight,
                    False,
                )
                * self.scaling[adapter]
            )

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                result = self.base_layer(x, *args, **kwargs)
                # As per Tim Dettmers, for 4bit, we need to defensively clone here.
                # The reason is that in some cases, an error can occur that backprop
                # does not work on a manipulated view. This issue may be solved with
                # newer PyTorch versions but this would need extensive testing to be
                # sure.
                result = result.clone()

                for active_adapter in self.active_adapters:
                    if active_adapter not in self.lora_A.keys():
                        continue
                    lora_A = self.lora_A[active_adapter]
                    lora_B = self.lora_B[active_adapter]
                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        x = x.to(lora_A.weight.dtype)

                    if self.qa:
                        x = self.qa_pool(x) * self.block_size

                    output = lora_B(lora_A(dropout(x)))
                    if requires_conversion:
                        output = output.to(expected_dtype)
                    output = output * scaling
                    result += output

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "lora." + rep

    class QALinear4bit(torch.nn.Linear):
        def __init__(self, input_features, output_features, bias=True, compute_dtype=None, compress_statistics=True, quant_type='fp4', device=None):
            super().__init__(input_features, output_features, bias, device)
            self.weight = torch.nn.Parameter(self.weight.data)
            self.compute_dtype = compute_dtype
            self.compute_type_is_set = False

        def set_compute_type(self, x):
            if x.dtype in [torch.float32, torch.bfloat16]:
                # the input is in a dtype that is safe to compute in, we switch
                # to this type for speed and stability
                self.compute_dtype = x.dtype
            elif x.dtype == torch.float16:
                # we take the compoute dtype passed into the layer
                if self.compute_dtype == torch.float32 and (x.numel() == x.shape[-1]):
                    # single batch inference with input torch.float16 and compute_dtype float32 -> slow inference when it could be fast
                    # warn the user about this
                    warnings.warn(f'Input type into Linear4bit is torch.float16, but bnb_4bit_compute_type=torch.float32 (default). This will lead to slow inference.')
                    warnings.filterwarnings('ignore', message='.*inference.')
                if self.compute_dtype == torch.float32 and (x.numel() != x.shape[-1]):
                    warnings.warn(f'Input type into Linear4bit is torch.float16, but bnb_4bit_compute_type=torch.float32 (default). This will lead to slow inference or training speed.')
                    warnings.filterwarnings('ignore', message='.*inference or training')

        def _save_to_state_dict(self, destination, prefix, keep_vars):
            raise NotImplementedError("Saving a merged quantized checkpoint is not supported for QALora")

        def forward(self, x: torch.Tensor):
            # weights are cast automatically as Int8Params, but the bias has to be cast manually
            if self.bias is not None and self.bias.dtype != x.dtype:
                self.bias.data = self.bias.data.to(x.dtype)

            if getattr(self.weight, 'quant_state', None) is None:
                raise ValueError(
                    f"No quantization state found for weight of layer {self.__class__.__name__}. "
                )
            
            if not self.compute_type_is_set:
                self.set_compute_type(x)
                self.compute_type_is_set = True

            inp_dtype = x.dtype
            if self.compute_dtype is not None:
                x = x.to(self.compute_dtype)

            bias = None if self.bias is None else self.bias.to(self.compute_dtype)
            c = (127 / self.weight.quant_state[0]).view(self.weight.shape[0], self.weight.shape[1]//self.weight.quant_state[3]).unsqueeze(2).expand(-1, -1, 64).reshape(self.weight.shape[0], self.weight.shape[1])
            out = x @ (self.weight / c).T.to(self.compute_dtype) + bias 

            out = out.to(inp_dtype)

            return out