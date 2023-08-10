from transformers import AutoTokenizer, AutoModel
from .chatglm_local.modeling_chatglm import ChatGLMForConditionalGeneration

def load_llm_model():
    tokenizer = AutoTokenizer.from_pretrained("/home/hmp/Miraclemarvel55_RLHF/chatglm_6b", trust_remote_code=True)
    if "cuda" in action_device:
        model = ChatGLMForConditionalGeneration.from_pretrained("/home/hmp/Miraclemarvel55_RLHF/chatglm_6b", trust_remote_code=True)
        model = model.half().cuda(action_device) # half for gpu only
    elif "cpu" == action_device:
        model = ChatGLMForConditionalGeneration.from_pretrained("/home/hmp/Miraclemarvel55_RLHF/chatglm_6b", trust_remote_code=True).bfloat16()

    # 只更新embedding
    model.requires_grad_(False)
    model.transformer.word_embeddings.requires_grad_(True)
    return model, tokenizer