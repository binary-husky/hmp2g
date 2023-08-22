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

def load_llm_model(full_head=True, tokenizer=True, device=None):
    from .foundation import AlgorithmConfig
    if full_head:
        from .chatglm1_modify.modeling_chatglm import ChatGLMForConditionalGeneration as Model
    else:
        from .chatglm1_modify.modeling_chatglm import ChatGLMModel as Model
    path = AlgorithmConfig.model_path
    if tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    else:
        tokenizer = None
    model = Model.from_pretrained(path, trust_remote_code=True)
    model = model.half().to(device) # half for gpu only

    if full_head:
        # 只更新embedding
        training_para_list = []
        model.requires_grad_(False)
        model.lm_head.requires_grad_(True)
        training_para_list.extend(list(model.lm_head.parameters()))
        for i, layer in enumerate(model.transformer.layers):
            if i < 2:
                layer.requires_grad_(True)
                training_para_list.extend(list(layer.parameters()))
        model.training_para_list = training_para_list
        # model.transformer.word_embeddings.requires_grad_(True)
    else:
        layers_keep = len(model.layers)//14
        # layers_keep = 3
        model.layers = model.layers[:layers_keep]
        # solve RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'
        model = model.half().cuda(device) # half for gpu only

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
        from .bridge_llm import load_llm_model
        self.model, _ = load_llm_model(full_head=False, device=device)
        self.output_linear = nn.Linear(self.model.hidden_size, 1, device=self.model.device, dtype=self.model.dtype)
        self.dtype = self.model.dtype
        self.device = self.model.device

    def forward(self, **kwargs):
        output = self.model(**kwargs)
        values = torch.tanh(self.output_linear(output.last_hidden_state))
        return values.transpose(0, 1).squeeze(-1)
