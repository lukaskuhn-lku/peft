import os

import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -*- coding: utf-8 -*-
"""Finetune-opt-bnb-peft.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jCkpikz0J2o20FBQmYmAGdiKmJGOMo-o

## Fine-tune large models using 🤗 `peft` adapters, `transformers` & `bitsandbytes`

In this tutorial we will cover how we can fine-tune large language models using the very recent `peft` library and `bitsandbytes` for loading large models in 8-bit.
The fine-tuning method will rely on a recent method called "Low Rank Adapters" (LoRA), instead of fine-tuning the entire model you just have to fine-tune these adapters and load them properly inside the model.
After fine-tuning the model you can also share your adapters on the 🤗 Hub and load them very easily. Let's get started!

### Install requirements

First, run the cells below to install the requirements:
"""


"""### Model loading

Here let's load the `opt-6.7b` model, its weights in half-precision (float16) are about 13GB on the Hub! If we load them in 8-bit we would require around 7GB of memory instead.
"""


free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
max_memory = f"{free_in_GB-2}GB"

n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-350m",
    max_memory=max_memory,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

"""### Post-processing on the model

Finally, we need to apply some post-processing on the 8-bit model to enable training, let's freeze all our layers, and cast the layer-norm in `float32` for stability. We also cast the output of the last layer in `float32` for the same reasons.
"""

print(model)

for param in model.parameters():
    param.requires_grad = False  # freeze the model - train adapters later
    if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)

# model.gradient_checkpointing_enable()  # reduce number of stored activations
# model.model.decoder.project_in = lambda x: x.requires_grad_(True)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


model.lm_head = CastOutputToFloat(model.lm_head)

"""### Apply LoRA

Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.
"""


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM",
    quantization_awareness=True,
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# Verifying the datatypes.
dtypes = {}
for _, p in model.named_parameters():
    dtype = p.dtype
    if dtype not in dtypes:
        dtypes[dtype] = 0
    dtypes[dtype] += p.numel()
total = 0
for k, v in dtypes.items():
    total += v
for k, v in dtypes.items():
    print(k, v, v / total)

"""### Training"""

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=0,
        max_steps=2,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# from huggingface_hub import notebook_login

# notebook_login()

# model.push_to_hub("ybelkada/opt-6.7b-lora", use_auth_token=True)

"""## Load adapters from the Hub

You can also directly load adapters from the Hub using the commands below:
"""

# import torch
# from peft import PeftModel, PeftConfig
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# peft_model_id = "ybelkada/opt-6.7b-lora"
# config = PeftConfig.from_pretrained(peft_model_id)
# model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
# tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
#
## Load the Lora model
# model = PeftModel.from_pretrained(model, peft_model_id)
#
# """## Inference
#
# You can then directly use the trained model or the model that you have loaded from the 🤗 Hub for inference as you would do it usually in `transformers`.
# """
#
batch = tokenizer("Two things are infinite: ", return_tensors="pt")

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
model.eval()
model = model.merge_and_unload()

with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=50)

print("\n\n", tokenizer.decode(output_tokens[0], skip_special_tokens=True))
# model.save('./test.pt')

# """As you can see by fine-tuning for few steps we have almost recovered the quote from Albert Einstein that is present in the [training data](https://huggingface.co/datasets/Abirate/english_quotes)."""
