import os, time, torch, json, shutil
import numpy as np
from UTIL.colorful import *
from config import GlobalConfig
from UTIL.tensor_ops import repeat_at
from ALGORITHM.common.rl_alg_base import RLAlgorithmBase
from ALGORITHM.common.logit2act import Logit2Act
from .temp import sample_good_QA_from_turns, get_log_prob, get_log_probs_with_input_ids
from .temp import RewardBySimilarity, Critic, gae_vectorize
from pathlib import Path
from torch.optim import Adam

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



class LLM_Foundation(RLAlgorithmBase):
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        from .bridge_llm import load_llm_model
        self.n_agent = n_agent
        self.data_set = json.loads(Path('/home/hmp/hmp2g/ALGORITHM/llm/profile_instance.json').read_text(encoding="utf8"))
        self.llm_model, self.tokenizer = load_llm_model()
        self.logic_processor = Logit2Act()
        self.reward_model = RewardBySimilarity(device=GlobalConfig.device)
        self.critic = Critic(device=GlobalConfig.device)
        optimize_params = list(self.llm_model.transformer.word_embeddings.parameters())+list(self.critic.parameters())
        self.optimizer = Adam(optimize_params, lr=1e-4, eps=1e-3)

    def interact_with_env(self, StateRecall):
        '''
            Interfacing with marl, standard method that you must implement
            (redirect to shell_env to help with history rolling)
        '''
        from .temp import generate_inputs
        action_device = GlobalConfig.device

        for i in range(len(self.data_set)):
            chat = self.data_set[i]

        good_qa = sample_good_QA_from_turns(chat)
        good_answers = chat[-1]["好答"]
        bad_answers = chat[-1]["坏答"]
        
        # r = random.randint(1, 5)
        if np.random.rand() < 0.5:
            query = good_qa[-1][0]
            history = good_qa[:-1]
            inputs, gen_len, prompt = generate_inputs(self.tokenizer, query=query, history=history)
            input_ids = inputs["input_ids"].to(action_device)
            num_beams, num_return_sequences = 1, 1 # 3, 2 # set bigger if you have bigger compute memory
            assert num_beams >= num_return_sequences, "candidates num should greater than returns num"
            max_new_tokens = 8
            gen_method = "greedy_search" if num_beams == 1 else "beam_search" 
            generate_ = self.llm_model.generate(input_ids=input_ids, do_sample=False, num_beams=num_beams, max_new_tokens=max_new_tokens,
                                num_return_sequences=num_return_sequences, use_cache=True, num_beam_groups=1, output_scores=True,
                                output_hidden_states=False, return_dict_in_generate=True)
            sequences = generate_.sequences
            
            log_probs = get_log_prob(generated_outputs=generate_, input_ids=input_ids, gen_method=gen_method)
            gen_texts = self.tokenizer.batch_decode(sequences[:, input_ids.shape[1]:])
            out_texts = self.tokenizer.batch_decode(sequences)
            # print(query, qa_logs[query], sep="\n")
            print(query, gen_texts, sep="\n")
        else:
            # 将目标句直接用RL提升或降低它的概率，得到类似finetune的效果
            query = ""
            inputs, gen_len, prompt = generate_inputs(self.tokenizer, query=query, history=good_qa) # '[Round 0]\n问：你的主人是谁？\n答：'
            input_ids = inputs["input_ids"].to(action_device)
            sequences = input_ids
            with torch.no_grad(): # 此处不需要梯度，后面PPO会二次计算
                log_probs = get_log_probs_with_input_ids(self.llm_model, input_ids, gen_max_len=gen_len)    # gen_len是good_qa输出的长度
            gen_texts = [good_qa[-1][1]]    # 回复样本y
            out_texts = self.tokenizer.batch_decode(sequences)
            # print("目标句直接用RL提升它的概率：", out_texts)

        # compute reward for generated sequences
        reward = self.reward_model(gen_texts=gen_texts, good_answers=good_answers, bad_answers=bad_answers).unsqueeze(1) # 获取终末稀疏奖励
        assert reward.shape == (len(gen_texts), 1), "need unsqueeze for next scatter_"
        rewards = torch.zeros_like( sequences, dtype=reward.dtype, device=reward.device)    # 准备一个长奖励向量，准备用gae奖励均摊
        # pad_id =   # 
        masks = (sequences!=self.tokenizer.convert_tokens_to_ids("<pad>")).long().to(GlobalConfig.device)  # mask 掉 pad
        final_position = masks.sum(dim=-1)-1
        index=final_position.unsqueeze(-1)
        rewards.scatter_(dim=1, index=index, src=reward)    # 把奖励放置到句子的最后一个token输出的地方 [0,0,0,0,...,reward,0,0,...,0,0]
        # 确保都放到values所在的device
        rewards = torch.tensor(rewards, dtype=self.critic.dtype, device=self.critic.device)
        masks = masks.to(self.critic.device)

        torch.cuda.empty_cache()
        self.ppo(ppo_epochs=5, states= sequences,log_probs=log_probs, rewards=rewards, masks=masks, clip_param=0.2)

        action = np.zeros(shape=(1, 1))
        return action, StateRecall

    def ppo(self, ppo_epochs, states, log_probs, rewards, masks, clip_param):
        for ppo_epoch in range(ppo_epochs):
            # compute new log probs
            new_log_probs = get_log_probs_with_input_ids(self.llm_model, states, log_probs.shape[1])
            entropy = 0 # 暂时不需要熵的约束
            # compute value
            # 到奖励模型和值函数模型的输入可以是一样的都是生成的序列。
            # 生成序列同时包括state和next action
            # prepare input for critic model
            input_ids_critic = states.to(GlobalConfig.device)
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
            loss = 0.5 * (critic_loss + actor_loss) - 0.001 * entropy
            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("loss", loss.item())
