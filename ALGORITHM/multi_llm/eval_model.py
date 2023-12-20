from config import GlobalConfig
from transformers import AutoTokenizer, AutoModel
from UTIL.tensor_ops import repeat_at, _2tensor, scatter_righthand
from ALGORITHM.llm.temp import tokenize_qa
import torch
from peft import get_peft_config, get_peft_model, LoraConfig, PeftModel
model_path = "/home/hmp/Miraclemarvel55_RLHF/chatglm_6b"
lora_path = 'RESULT/llm_trainer_critical/llm_model/saved'
# path = 'RESULT/LLM'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
model = PeftModel.from_pretrained(model, lora_path)
model = model.half().to(GlobalConfig.device) # half for gpu only
model.requires_grad_(False)
max_gen_tokens = 128

while True:
    q = input('>>')
    token_qa_prompt, gen_len, prompt = tokenize_qa(tokenizer, query=q, history=[])
    input_ids = torch.Tensor(token_qa_prompt['input_ids']).to(GlobalConfig.device)
    num_beams, num_return_sequences = 1, 1
    assert num_beams >= num_return_sequences, "candidates num should greater than returns num"
    gen_method = "greedy_search" if num_beams == 1 else "beam_search" 
    model_result = model.generate(input_ids=input_ids, do_sample=False, num_beams=num_beams, max_new_tokens=max_gen_tokens,
                        num_return_sequences=num_return_sequences, use_cache=True, num_beam_groups=1, output_scores=True,
                        output_hidden_states=False, return_dict_in_generate=True)
    sequences = model_result.sequences
    
    gen_texts = tokenizer.batch_decode(sequences)
    print(gen_texts[0])
