"""
classes for ChatGLM RLHF
ChatGLMCritic model
Action model is ChatGLM, 所以可省略
Reward model

"""
import torch
from .chatglm1_modify.modeling_chatglm import ChatGLMModel
from torch import nn
from transformers import BertTokenizer, BertModel
import numpy as np
from functools import partial


"""
一样的原因，不需要再把生成的token ids转成文字在再转到目标ids，
所以也用chatglm直接做基模型，
只是这里只取最后的token算出对整句生成的奖励分数，具体取哪个位置可以
后续在代码里面指定，比如用torch.gather
""" 

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask): # attention_mask 主要是考虑每个句子的长度，用0去pad短句子
    token_embeddings = model_output[0]  # last_hidden_state: contains all token embeddings, (batch_size, sentence_len, core_dim)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    # token_embeddings * input_mask_expanded 将pad部分清零, torch.sum将各个句子token_embeddings求和，input_mask_expanded.sum(1)是各个句子的长度
def jaccard(s1, s2):
    """
    可能有字符串重合但是语义不一致问题，
    TODO 可以用多阶的jaccard来解决
    """
    assert len(s1)+len(s2)>0
    s1 = set(s1)
    s2 = set(s2)
    s_or = s1 | s2
    s_and = s1 & s2
    jaccard_distance = len(s_and)/len(s_or)
    return jaccard_distance

# 基于和good_answers和bad_answers比较的Reward模型，具有通用性和易学习，基本上都是基于Bert余弦值
class RewardBySimilarity(nn.Module):

    def __init__(self, device="cpu") -> None:
        super().__init__()
        # Load model from HuggingFace Hub
        tokenizer = BertTokenizer.from_pretrained('shibing624/text2vec-base-chinese')   # Bert比GPT具有更好的下文关联能力
        model = BertModel.from_pretrained('shibing624/text2vec-base-chinese')           # Bert比GPT具有更好的下文关联能力
        model.eval()
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, gen_texts_batch, good_answers_batch, bad_answers_batch, weight_for_cos_and_jaccard = [0.5, 0.5]):
        reward_batch = []
        for gen_texts, good_answers, bad_answers in zip(gen_texts_batch, good_answers_batch, bad_answers_batch):
            reward = self.forward_(gen_texts, good_answers, bad_answers, weight_for_cos_and_jaccard)
            reward_batch.append(reward)
        return torch.stack(reward_batch)
    
    def forward_(self, gen_texts,  good_answers, bad_answers, weight_for_cos_and_jaccard):
        gen_texts = [gen_texts]
        examples = good_answers + bad_answers
        example_num = len(examples)
        assert len(gen_texts)>0 and example_num>0
        reward_direction = torch.ones(example_num, device=self.model.device)
        reward_direction[len(good_answers):] = -1   # 好样本 reward_direction 1, 坏样本 reward_direction -1
        sentences = gen_texts + examples    # 好样本 + 坏样本
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, return_tensors='pt')    # 将所有样本转为token（经过pad）
        ids = self.tokenizer.batch_encode_plus(sentences, add_special_tokens=False)["input_ids"]    # 没有pad的tokenID
        # temporary truncate position_ids
        batch_size, max_seq_len = encoded_input["input_ids"].shape
        if max_seq_len > self.model.config.max_position_embeddings:
            encoded_input["position_ids"] = torch.arange(max_seq_len).expand((1, -1)).repeat(batch_size, 1)
            encoded_input["position_ids"] = encoded_input["position_ids"]/max_seq_len*self.model.config.max_position_embeddings
            encoded_input["position_ids"] = encoded_input["position_ids"].floor().long()
        # Compute token embeddings
        with torch.no_grad():
            encoded_input = encoded_input.to(self.model.device)
            model_output = self.model(**encoded_input)  # 这里是BERT模型（中文）
        # Perform pooling. In this case, max pooling.
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])   # 把各个句子的 全句子embedding 平均，得到整个句子的整体语义嵌入
        gen_text_vecs = sentence_embeddings[:len(gen_texts)]    # 取出GLM生成的句子的整体语义嵌入
        answers_vecs = sentence_embeddings[len(gen_texts):]     # 取出好答案和坏答案的整体语义嵌入
        reward_ = []
        for i in range(gen_text_vecs.shape[0]): # 对于每一个GLM生成的句子
            gen_text_vecs_ = gen_text_vecs[i:i+1]   # 等价于select&unsqueeze()
            # 用一下广播计算cos
            coses = torch.cosine_similarity(gen_text_vecs_, answers_vecs, dim=1)    # 与每个好答案和坏答案的余弦相似性
            # 余弦截断
            coses[(coses<0)] = 0    # 越相似，cos约接近1； 小于0的话，说明差异很大
            # 计算 jaccard距离 （没看懂，这个作用？）
            jaccard_s1 = partial(jaccard, ids[i])
            jaccards = torch.tensor(np.vectorize(jaccard_s1)(np.array(ids[-len(examples):], dtype=object)), dtype=coses.dtype, device=coses.device)
            similarity = weight_for_cos_and_jaccard[0]*coses + weight_for_cos_and_jaccard[1]*jaccards
            value, index = similarity.max(dim=-1)   # GLM生成的句子 和 哪个参考答案最相似，相似性 value [0, 1]
            reward_.append(value*reward_direction[index])   # 如果参考答案是good，则给与终末稀疏奖励 +value； 否则给与终末稀疏奖励 -value
        reward = torch.stack(reward_)
        return reward   # 返回终末稀疏奖励

def test_reward_by_similarity():
    reward_model = RewardBySimilarity()
    reward = reward_model()
    print(reward)



def test_reward():
    # with torch.no_grad():
    # input_ids_RM =  sequences.to(RM_device)
    # rewards_ = reward_model(input_ids = input_ids_RM)
    # # 由于只对最后的整句进行reward，所以只有最后一个action后的state有reward
    # rewards = torch.zeros_like( sequences, dtype=rewards_.dtype, device=rewards_.device)
    # pad_id = tokenizer.convert_tokens_to_ids("<pad>")
    # masks = ( sequences!=pad_id).long().to(RM_device)

    # final_position = masks.sum(dim=-1)-1
    # index=final_position.unsqueeze(-1)
    # reward = rewards_.gather(dim=1, index=index)
    # rewards.scatter_(dim=1, index=index, src=reward)
    pass


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import random
def sample_good_QA_from_turns(turns):
    history = [ [turn["question"], random.choice(turn["good_ans"])] for turn in turns ]
    return history

def tokenize_qa(tokenizer, query='', history=[]):
    assert query or history, "query and history cannot both empty"

    prompt = ""
    for i, (old_query, response) in enumerate(history):
        if i==len(history)-1 and query == "":
            prompt += "[Round {}]\n问：{}\n答：".format(i, old_query)
        else:
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
    if query:
        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        
    inputs = tokenizer([prompt], return_tensors="pt")
    gen_len = 0
    if query=="":
        # query为空代表history的最后一个回答是目标答案
        prompt += history[-1][1]
        last_response_encode = tokenizer.encode(history[-1][1], return_tensors="pt", add_special_tokens=False)
        if last_response_encode[0, 0] == 5:
            last_response_encode = last_response_encode[:, 1:]
            # TODO batch化
        eops = torch.zeros_like(last_response_encode[:, :1])+tokenizer.convert_tokens_to_ids("<eop>")
        # TODO 后续用scatter来放到可能多个句子的带padding的正确位置，暂时先放到最后，因为现在只有一句
        last_response_encode = torch.cat([last_response_encode, eops], dim=-1)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], last_response_encode], dim=-1)
        gen_len = last_response_encode.shape[1]
    return inputs, gen_len, prompt

def tokenize_qa_old(tokenizer, query='', history=[]):
    assert query or history, "query and history cannot both empty"
    if not history:
        prompt = query
    else:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            if i==len(history)-1 and query == "":
                prompt += "[Round {}]\n问：{}\n答：".format(i, old_query)
            else:
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        if query:
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
    inputs = tokenizer([prompt], return_tensors="pt")
    gen_len = 0
    if query=="":
        # query为空代表history的最后一个回答是目标答案
        prompt += history[-1][1]
        last_response_encode = tokenizer.encode(history[-1][1], return_tensors="pt", add_special_tokens=False)
        if last_response_encode[0, 0] == 5:
            last_response_encode = last_response_encode[:, 1:]
            # TODO batch化
        eops = torch.zeros_like(last_response_encode[:, :1])+tokenizer.convert_tokens_to_ids("<eop>")
        # TODO 后续用scatter来放到可能多个句子的带padding的正确位置，暂时先放到最后，因为现在只有一句
        last_response_encode = torch.cat([last_response_encode, eops], dim=-1)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], last_response_encode], dim=-1)
        gen_len = last_response_encode.shape[1]
    return inputs, gen_len, prompt

def get_log_prob(generated_outputs, input_ids, gen_method = "greedy_search"):
    # beam_search generate 给出来的scores就是log_prob了，所以直接gather获取即可
    gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1]:] 
    # let's stack the logits generated at each step to a tensor
    # 要小心greedy search 拿到的是score，需要再log_softmax
    # 而beam_search 拿到的已经是log_softmax了
    scores = torch.stack(generated_outputs.scores, dim=1)
    # if scores.max() >0 :
    #     gen_method = "greedy_search"
    if gen_method == "beam_search":
        log_prob_stacked = scores
    else:
        log_prob_stacked = torch.stack(generated_outputs.scores, dim=1).log_softmax(dim=-1)
    # now we need to collect the log_prob of the generated token # we need to add a dummy dim in the end to make gather work 
    log_prob = torch.gather(log_prob_stacked, 2, gen_sequences[:, :, None]).squeeze(-1)
    return log_prob

def get_log_probs_with_input_ids(model, input_ids, gen_max_len):  # 自回归获取对应的输入logits的logprob，用于后续的强化
    # input_ids torch.Size([3, 23])
    model_inputs = model.prepare_inputs_for_generation(input_ids)
    output = model(**model_inputs)  # 将已经生成的序列放进去计算，再次计算得到目标action也就是后续字符的概率或者log_prob值
    # output.logits.shape torch.Size([3, 23, 130528])
    # get bootstrap prob ! 
    # out.logits
    
    # 对齐
    seq_dimension = 1
    align_logit_bootstrap = torch.cat((output.logits[:, 0:1, :] * 0, output.logits[:, :-1, :]), axis=seq_dimension)


    logits = align_logit_bootstrap.log_softmax(dim=-1)[:, -gen_max_len:] # 比先softmax再log好,复杂度减小，并且解决些nan问题
    new_log_probs = logits.gather(dim=-1, index=input_ids[:, -gen_max_len:].unsqueeze(-1)).squeeze(-1)
    return new_log_probs


from functools import lru_cache
@lru_cache
def get_decay_up_matrix_T(dtype=torch.float, device="cpu", max_length = 2048, gamma=0.99, tau=0.95):
    # 生成衰减矩阵
    decay = gamma*tau
    decay_row = torch.ones(max_length, dtype=dtype, device=device)*decay
    decay_row[0] = 1
    decay_row_cross_time = decay_row.cumprod(dim=-1)
    assert decay_row_cross_time.sign().min() == 0
    decay_up_matrix = torch.zeros((max_length, max_length), dtype=dtype, device=device)
    for i in range(max_length):
        decay_row = decay_row_cross_time.roll(i)
        decay_row[:i] = 0 # 确保看不见前面的
        decay_up_matrix[i] = decay_row
    decay_up_matrix_T = decay_up_matrix.T# 先进行转置，因为后面需要用到矩阵乘法
    return decay_up_matrix_T


def gae_vectorize(values, rewards, masks=None):
    """
        values:表示各个时间步状态的状态值。shape:batch_size,sequence_length
        rewards:表示各个时间步做出的动作的奖励，对于gpt当前动作也是动作对应的下一状态。所以shape和values一样
                # 注意这里的rewards表示当前动作状态的reward
        masks:由于是要对生成的actions做gae，也就是泛化优势估计，
                # 所以类似以往的mask只需要对padding进行mask，
                # 因为padding的delta会被放入加权计算，而action前面的delta，
                # 由于生成的衰减矩阵就是上三角的，自然就看不到前面的。
                # 0表示mask， 1表示需要的。
    """
    action_rewards = rewards.roll(-1) # 当前状态的动作的奖励是下一个状态出现时给出的，而奖励是基于状态计算的，所以需要shift一个时间步回去
    # 为了学到最后输出的<eop>,所以给最后的状态赋予一个rewards试试
    action_rewards = (action_rewards+rewards)/2 # 将奖励分配到最后两步

    values_estimator_1_order = action_rewards + values.roll(-1) # 这里要注意roll是循环的，所以最后一位的值可能不能用
    deltas = values_estimator_1_order - values  #必须要action+下一个时刻的值函数减去当前值函数，这是表示当前action的优势
    # get weight matrix
    decay_up_matrix_T = get_decay_up_matrix_T(dtype=deltas.dtype, device= deltas.device)
    # 计算gae
    max_goal_length = deltas.shape[-1]
    sub_decay_up_matrix_T = decay_up_matrix_T[:max_goal_length, :max_goal_length]
    if masks is not None:
        deltas = deltas * masks
    gae = deltas.matmul(sub_decay_up_matrix_T.to(deltas.device))
    assert gae.shape == deltas.shape
    return gae
