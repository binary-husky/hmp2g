import os, time, torch, json, random
import numpy as np
from UTIL.colorful import *
from config import GlobalConfig
from UTIL.tensor_ops import repeat_at
from ALGORITHM.common.rl_alg_base import RLAlgorithmBase
from ALGORITHM.common.logit2act import Logit2Act
from .temp import sample_good_QA_from_turns, get_log_prob, get_log_probs_with_input_ids
from UTIL.tensor_ops import repeat_at, _2tensor, scatter_righthand
from .temp import RewardBySimilarity, gae_vectorize
from .bridge_llm import ChatGLMCritic
from pathlib import Path
from torch.optim import Adam
from .temp import tokenize_qa
import torch.nn as nn
import torch.nn.functional as F

class AlgorithmConfig:
    '''
        AlgorithmConfig: This config class will be 'injected' with new settings from json.
        (E.g., override configs with ```python main.py --cfg example.jsonc```)
        (please see UTIL.config_args to find out how this advanced trick works out.)
    '''
    # configuration, open to jsonc modification
    gamma = 0.99
    tau = 0.95
    train_traj_needed = 512
    dataset_path = './ALGORITHM/llm/profile_instance.json.insert.json'
    model_path = "/home/hmp/Miraclemarvel55_RLHF/chatglm_6b"
    max_source_length = 64
    mini_batch_size = 18
    max_gen_tokens = 64
    debug = False

    device_override = "cuda:0,1"
    device_main = 'cuda:1'
    device_secondary = 'cuda:0'

    save_step = 50


def override_cuda_settings(AlgorithmConfig):
    # change Local cuda settings according to AlgorithmConfig
    if AlgorithmConfig.device_override != "no-override":
        assert GlobalConfig.device == 'cpu', "please set GlobalConfig.device=cpu if you want to use different GPUs for different teams"
        # reflesh the cuda setting inherited from main.py
        GlobalConfig.device = AlgorithmConfig.device_override
        from main import pytorch_gpu_init; pytorch_gpu_init(GlobalConfig)
        # reflesh the cached cuda setting in tensor_ops
        from UTIL.tensor_ops import cuda_cfg; cuda_cfg.read_cfg()


class LLM_Foundation(RLAlgorithmBase):
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        override_cuda_settings(AlgorithmConfig)
        from .bridge_llm import load_llm_model
        self.n_agent = n_agent
        self.data_set = json.loads(Path(AlgorithmConfig.dataset_path).read_text(encoding="utf8"))
        self.flat_samples(self.data_set)
        self.llm_model, self.tokenizer = load_llm_model(full_head=True, device=AlgorithmConfig.device_main)

        self.llm_model_ref, _ = load_llm_model(full_head=True, tokenizer=False, device=AlgorithmConfig.device_secondary)
        self.llm_model_ref.requires_grad_(False)
        
        self.reward_model = RewardBySimilarity(device=AlgorithmConfig.device_main)
        self.critic = ChatGLMCritic(device=AlgorithmConfig.device_main)
        optimize_params = self.llm_model.training_para_list + list(self.critic.parameters())
        self.optimizer = Adam(optimize_params, lr=1e-4, eps=1e-3)
        self.logic_processor = Logit2Act()

        self.data_set = json.loads(Path(AlgorithmConfig.dataset_path).read_text(encoding="utf8"))
        self.mcv = mcv

    def flat_samples(self, data_set):
        self.data_set_collection = []
        for i in range(len(data_set)):
            question = data_set[i][0]['question']
            good_ans = data_set[i][0]['good_ans']
            bad_ans = data_set[i][0]['bad_ans']
            self.data_set_collection.append({
                'question': question,
                'good_ans_demos': good_ans,
                'bad_ans_demos': bad_ans,
                'score': 1
            })

        self.data_set_collection = np.array(self.data_set_collection, dtype=object)
        return 
    
    def sample_qa(self):
        if not hasattr(self, 'sampler'):
            from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
            big_batch_size = len(self.data_set_collection)
            self.sampler = BatchSampler(SubsetRandomSampler(range(big_batch_size)), AlgorithmConfig.mini_batch_size, drop_last=True)
            self.sampler = iter(self.sampler)

        try: 
            indices = next(self.sampler)
        except StopIteration:
            from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
            big_batch_size = len(self.data_set_collection)
            self.sampler = BatchSampler(SubsetRandomSampler(range(big_batch_size)), AlgorithmConfig.mini_batch_size, drop_last=True)
            self.sampler = iter(self.sampler)
            indices = next(self.sampler)

        samples = self.data_set_collection[indices]
        pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")

        query = [s['question'] for s in samples]
        good_ans_demos = [s['good_ans_demos'] for s in samples]
        bad_ans_demos = [s['bad_ans_demos'] for s in samples]

        res = {
            "token_qa_prompt_array": [],
            "token_q_prompt_array": [],
            "tokenlen_qa_prompt_array": [],
            "tokenlen_q_prompt_array": [],
            "good_ans_demos": [],
            "bad_ans_demos": [],
            "good_ans_demo": [],
        }
        for i, q in enumerate(query):
            good_ans_demo = random.choice(good_ans_demos[i])
            token_qa_prompt, gen_len, prompt = tokenize_qa(self.tokenizer, query="", history=[[q, good_ans_demo]])
            token_qa_prompt = token_qa_prompt['input_ids'].tolist()[0]
            res["token_qa_prompt_array"].append(token_qa_prompt)
            res["tokenlen_qa_prompt_array"].append(len(token_qa_prompt))
            token_q_prompt, gen_len, prompt = tokenize_qa(self.tokenizer, query=q, history=[])
            token_q_prompt = token_q_prompt['input_ids'].tolist()[0]
            res["token_q_prompt_array"].append(token_q_prompt)
            res["tokenlen_q_prompt_array"].append(len(token_q_prompt))
            res["good_ans_demo"].append(good_ans_demo)
            res["good_ans_demos"].append(good_ans_demos[i])
            res["bad_ans_demos"].append(bad_ans_demos[i])

        # pad
        def pad(arr):
            len_list = [len(token_qa_prompt) for i, token_qa_prompt in enumerate(arr)]
            max_len = max(len_list)
            for i, token_qa_prompt in enumerate(arr):
                arr[i] = [pad_id] * (max_len - len_list[i]) + token_qa_prompt
            return np.array(arr)
        
        res["token_qa_prompt_array"] = pad(res["token_qa_prompt_array"])
        res["token_q_prompt_array"] = pad(res["token_q_prompt_array"])
        res["tokenlen_q_pad"] = res["token_q_prompt_array"].shape[-1]
        res["tokenlen_qa_pad"] = res["token_qa_prompt_array"].shape[-1]
        res["tokenlen_a_pad"] = res["tokenlen_qa_pad"] - res["tokenlen_q_pad"]
        return res

    def interact_with_env(self, StateRecall):
        '''
            Interfacing with marl, standard method that you must implement
            (redirect to shell_env to help with history rolling)
        '''
        from .temp import tokenize_qa

        res = self.sample_qa()

        with torch.no_grad(): # 此处不需要梯度，后面PPO会二次计算
            input_ids = _2tensor(res['token_q_prompt_array']).to(AlgorithmConfig.device_main)
            num_beams, num_return_sequences = 1, 1 # 3, 2 # set bigger if you have bigger compute memory
            assert num_beams >= num_return_sequences, "candidates num should greater than returns num"
            # max_new_tokens = 8
            gen_method = "greedy_search" if num_beams == 1 else "beam_search" 
            model_result = self.llm_model.generate(input_ids=input_ids, do_sample=False, num_beams=num_beams, max_new_tokens=AlgorithmConfig.max_gen_tokens,
                                num_return_sequences=num_return_sequences, use_cache=True, num_beam_groups=1, output_scores=True,
                                output_hidden_states=False, return_dict_in_generate=True)
            sequences1 = model_result.sequences
            
            log_probs1 = get_log_prob(generated_outputs=model_result, input_ids=input_ids, gen_method=gen_method)
            gen_texts1 = self.tokenizer.batch_decode(sequences1)
            print(gen_texts1[0])


            # 强化finetune模拟
            input_ids = _2tensor(res['token_qa_prompt_array']).to(AlgorithmConfig.device_main)
            gen_texts_sft_sim = res["good_ans_demo"]
            log_probs_sft_sim = get_log_probs_with_input_ids(self.llm_model, input_ids, res["tokenlen_a_pad"])    # gen_len是good_qa输出的长度
            sequences_sft_sim = input_ids

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        gen_texts = gen_texts1 + gen_texts_sft_sim
        good_answers_batch = res['good_ans_demos'] + res['good_ans_demos']
        bad_answers_batch = res['bad_ans_demos'] + res['bad_ans_demos']


        clip_size = sequences_sft_sim.shape[-1]
        log_clip_size = log_probs_sft_sim.shape[-1]

        sequences = torch.cat((sequences1[..., :clip_size], sequences_sft_sim[..., :clip_size]), 0)
        log_probs = torch.cat((log_probs1[..., :log_clip_size], log_probs_sft_sim[..., :log_clip_size]), 0)

        # compute reward for generated sequences
        reward = self.reward_model(gen_texts_batch=gen_texts, good_answers_batch=good_answers_batch, bad_answers_batch=bad_answers_batch) # 获取终末稀疏奖励
        rewards, masks = self.place_reward(res["tokenlen_q_pad"], sequences, reward, self.tokenizer.convert_tokens_to_ids("<pad>"))
        torch.cuda.empty_cache()
        self.ppo(ppo_epochs=5, sequences=sequences, log_probs=log_probs, rewards=rewards, masks=masks, clip_param=0.2)
        action = np.zeros(shape=(1, 1))

        if StateRecall['Current-Obs-Step'][0] % AlgorithmConfig.save_step == 1: # AlgorithmConfig.save_step-1:
            self.save_model(out_dir=f'RESULT/{GlobalConfig.note}/llm_model')
        return action, StateRecall
    
    def save_model(self, out_dir):
        import shutil
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir,'saved')
        shutil.rmtree(model_path, ignore_errors=True)
        shutil.copytree(AlgorithmConfig.model_path, model_path)
        self.llm_model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

    def place_reward(self, len_of_query, sequences, reward, pad_id):
        rewards = torch.zeros_like( sequences, dtype=reward.dtype, device=reward.device) + np.nan
        rewards[sequences != pad_id] = 0
        rewards[..., :len_of_query] = np.nan
        masks = ~torch.isnan(rewards)
        # push reward forward
        sentence_end_point = (~torch.isnan(rewards)).sum(-1) + len_of_query - 1  # location where we should insearch the reward
        rewards = scatter_righthand(rewards, reward, sentence_end_point.unsqueeze(-1), check=True)    # insert reward
        rewards = torch.nan_to_num(rewards, 0)
        return rewards, masks


    def ppo(self, ppo_epochs, sequences, log_probs, rewards, masks, clip_param):
        for ppo_epoch in range(ppo_epochs):
            # compute new log probs
            new_log_probs = get_log_probs_with_input_ids(self.llm_model, sequences, log_probs.shape[1])
            with torch.no_grad():
                new_log_probs_ref = get_log_probs_with_input_ids(self.llm_model_ref, sequences.to(self.llm_model_ref.device), log_probs.shape[1])
            
            # entropy = 0 # 暂时不需要熵的约束
            # compute value
            # 到奖励模型和值函数模型的输入可以是一样的都是生成的序列。
            # 生成序列同时包括state和next action
            # prepare input for critic model
            input_ids_critic = sequences.to(AlgorithmConfig.device_main)
            values = self.critic(input_ids=input_ids_critic)
            # compute gae
            gae = gae_vectorize(values=values, rewards=rewards, masks=masks)
            advantages = gae[:, -log_probs.shape[-1]:].to(new_log_probs.device) # 根据参考输出的长度进行截断
            # 计算value的估计量的偏差作为actor loss
            # 以及ppo的actor_loss
            value_estimator_delta = advantages
            ratio = (new_log_probs - log_probs).exp()
            # print("reward", reward, "ratio:", ratio, sep="\n")
            if torch.isinf(ratio).any():
                break
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = value_estimator_delta.square().mean()

            kl_div_loss = F.kl_div(new_log_probs_ref.to(new_log_probs.device), new_log_probs, log_target=True, reduction="none").mean()

            loss = critic_loss + actor_loss + kl_div_loss * 0.001  # self.llm_model_ref
            self.mcv.rec(critic_loss.item(),'critic_loss')
            self.mcv.rec(actor_loss.item(),'actor_loss')
            self.mcv.rec(kl_div_loss.item(),'kl_div_loss')
            self.mcv.rec_show()
            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("loss", loss.item())
