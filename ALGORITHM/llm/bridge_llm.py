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

def load_llm_model():
    from .chatglm1_modify.modeling_chatglm import ChatGLMForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained("/home/hmp/Miraclemarvel55_RLHF/chatglm_6b", trust_remote_code=True)
    model = ChatGLMForConditionalGeneration.from_pretrained("/home/hmp/Miraclemarvel55_RLHF/chatglm_6b", trust_remote_code=True)
    model = model.half().to(GlobalConfig.device) # half for gpu only

    # 只更新embedding
    model.requires_grad_(False)
    model.transformer.word_embeddings.requires_grad_(True)
    return model, tokenizer
