import sys
import torch
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import random
from collections import defaultdict
import numpy as np
from config import GlobalConfig
from torch import nn
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
def load_static_model(device=None):
    from .foundation import AlgorithmConfig
    path = AlgorithmConfig.model_path
    from .chatglm1_modify.modeling_chatglm import ChatGLMForConditionalGeneration as Model
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = Model.from_pretrained(path, trust_remote_code=True)
    model = model.half().to(device) # half for gpu only
    model.training_para_list = None
    return model, tokenizer

def load_trainable_model(device=None):
    from .foundation import AlgorithmConfig
    path = AlgorithmConfig.model_path
    from .chatglm1_modify.modeling_chatglm import ChatGLMForConditionalGeneration as Model
    from peft import get_peft_config, get_peft_model, LoraConfig, PeftModel
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = Model.from_pretrained(path, trust_remote_code=True)
    model = model.half().to(device) # half for gpu only
    training_para_list = []
    model.requires_grad_(False)


    lora_path = 'RESULT/llm_trainer_critical/llm_model/saved'
    model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)

    # lora_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )
    # model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    for _, param in model.named_parameters():
        if param.requires_grad:
            training_para_list.append(param)

    model.training_para_list = training_para_list
    return model, tokenizer

def load_trainable_headless_model(device=None):
    from .foundation import AlgorithmConfig
    path = AlgorithmConfig.model_path
    from .chatglm1_modify.modeling_chatglm import ChatGLMModel as Model
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = Model.from_pretrained(path, trust_remote_code=True)
    model = model.half().to(device) # half for gpu only
    
    layers_keep = len(model.layers)//14
    # layers_keep = 3
    model.layers = model.layers[:layers_keep]
    # solve RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'
    model = model.half().cuda(device) # half for gpu only

    model.training_para_list = None
    return model, tokenizer

"""
critic 的词表最好和action模型的词表一样这样便于对每个生成的token进行打分，
不一致的词表会导致打分不对齐，所以选择用一样的模型但是加一下打分的输出
为了减小打分模型的大小，可以把原来的模型的layers缩减层数。
这样直接继承了，原来的token embedding
"""
class ChatGLMCritic(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.model, _ = load_trainable_headless_model(device=device)
        self.output_linear = nn.Linear(self.model.hidden_size, 1, device=self.model.device, dtype=self.model.dtype)
        self.dtype = self.model.dtype
        self.device = self.model.device

    def forward(self, **kwargs):
        output = self.model(**kwargs)
        values = torch.tanh(self.output_linear(output.last_hidden_state))
        return values.transpose(0, 1).squeeze(-1)
