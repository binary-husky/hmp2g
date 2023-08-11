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

def load_llm_model(full_head=True):
    if full_head:
        from .chatglm1_modify.modeling_chatglm import ChatGLMForConditionalGeneration as Model
    else:
        from .chatglm1_modify.modeling_chatglm import ChatGLMModel as Model
    path = "/home/hmp/Miraclemarvel55_RLHF/chatglm_6b"
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = Model.from_pretrained(path, trust_remote_code=True)
    model = model.half().to(GlobalConfig.device) # half for gpu only

    if full_head:
        # 只更新embedding
        model.requires_grad_(False)
        model.transformer.word_embeddings.requires_grad_(True)
    else:
        layers_keep = len(model.layers)//9
        layers_keep = 1
        model.layers = model.layers[:layers_keep]
        # solve RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'
        model = model.half().cuda(GlobalConfig.device) # half for gpu only

    return model, tokenizer

"""
critic 的词表最好和action模型的词表一样这样便于对每个生成的token进行打分，
不一致的词表会导致打分不对齐，所以选择用一样的模型但是加一下打分的输出
为了减小打分模型的大小，可以把原来的模型的layers缩减层数。
这样直接继承了，原来的token embedding
"""
class ChatGLMCritic(nn.Module):
    def __init__(self, device="cpu_float") -> None:
        super().__init__()
        from .bridge_llm import load_llm_model
        self.model, _ = load_llm_model(full_head=False)
        self.output_linear = nn.Linear(self.model.hidden_size, 1, device=self.model.device, dtype=self.model.dtype)
        self.dtype = self.model.dtype
        self.device = self.model.device

    def forward(self, **kwargs):
        output = self.model(**kwargs)
        values = torch.tanh(self.output_linear(output.last_hidden_state))
        return values.transpose(0, 1).squeeze(-1)

# def test_critic():
#     from transformers import AutoTokenizer, AutoModel
#     tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
#     critic = ChatGLMCritic()
#     input_ids = torch.tensor(tokenizer.encode("你好"), dtype=torch.long).unsqueeze(0)
#     input_ids = input_ids.repeat(2,1)
#     output = critic(input_ids=input_ids)  
#     print(output.shape)