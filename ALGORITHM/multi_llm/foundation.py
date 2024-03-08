import os, time, torch, json, random
import numpy as np
from UTIL.colorful import *
from config import GlobalConfig
from UTIL.tensor_ops import repeat_at
from ALGORITHM.common.rl_alg_base import RLAlgorithmBase
from ALGORITHM.common.logit2act import Logit2Act
from .temp import sample_good_QA_from_turns, get_log_prob, get_log_probs_with_input_ids
from UTIL.tensor_ops import repeat_at, _2tensor, scatter_righthand, my_view
from .temp import RewardBySimilarity, gae_vectorize
from .bridge_llm import ChatGLMCritic
from pathlib import Path
from torch.optim import Adam
from .temp import tokenize_qa
import torch.nn as nn
import torch.nn.functional as F
import void_terminal as vt
from .llm_util_base import VtTopicLauncher, TopicLauncher, RewardEvaluator, Reviser, SimpleDatasetManager

# vt.set_conf(key="API_KEY", value="6424e9d19e674092815cea1cb35e67a5")
# vt.set_conf(key="AZURE_ENDPOINT", value="https://rhtjjjjjj.openai.azure.com/")
# vt.set_conf(key="AZURE_ENGINE", value="qqwe")
# vt.set_conf(key="LLM_MODEL", value="azure-gpt-3.5")

vt.set_conf(key="API_KEY", value="fk195831-IdP9Pb3W6DCMUIbQwVX6MsSiyxwqybyS")
vt.set_conf(key="LLM_MODEL", value="api2d-gpt-3.5-turbo")

class Generator:
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        self.value = yield from self.gen

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
    max_source_length = 100
    max_gen_tokens = 100
    mini_batch_size = 16
    sft_batch_size = 16
    debug = False

    # device_override = "cuda:1"
    # device_launcher_llm = 'cuda:1'
    # device_main_llm = 'cuda:1'
    # device_reward_llm = 'cuda:1'

    device_override = "cuda:0,1"
    device_launcher_llm = 'cuda:0'
    device_main_llm = 'cuda:1'
    device_reward_llm = 'cuda:0'

    enable_rl = False
    enable_stf = True

    save_step = 50
    topics = [
        "房价过高", "内卷", "AI威胁", "量子计算",
        "房地产调控", "汇率", "字幕组", "网红",
        "GDP", "家庭", "视觉", "守望先锋",
        "金砖国家","俄罗斯","月球","美国中央情报局",
        "被骗到缅北", "平壤", "特斯拉", "未来生活"
    ]
    topic_launcher_prompt = f"针对“TOPIC_REPLACE”话题，提出一个引人深思的问题。限20字。"
    reward_model_prompt = f"问题：Q_REPLACE\n\n回答：A_REPLACE\n\n判断：以上文本的回答部分是否通俗，让中学学历的人也能看懂？仅回答“是”或“否”。"
    # reviser_prompt = 'Answer with Chinese, and you must use a metaphor and give a short answer. '   # 这个prompt建议使用英文
    reviser_prompt_input = "\n使用抬杠的语气否定该问题的正确性。将你的回复放入json的answer字段，40字以内。"


    dataset_size = 1000

def override_cuda_settings(AlgorithmConfig):
    # change Local cuda settings according to AlgorithmConfig
    if AlgorithmConfig.device_override != "no-override":
        assert GlobalConfig.device == 'cpu', "please set GlobalConfig.device=cpu if you want to use different GPUs for different teams"
        # reflesh the cuda setting inherited from main.py
        GlobalConfig.device = AlgorithmConfig.device_override
        from main import pytorch_gpu_init; pytorch_gpu_init(GlobalConfig)
        # reflesh the cached cuda setting in tensor_ops
        from UTIL.tensor_ops import cuda_cfg; cuda_cfg.read_cfg()

class LLM_Foundation(RLAlgorithmBase, VtTopicLauncher, RewardEvaluator, Reviser, SimpleDatasetManager):

    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        override_cuda_settings(AlgorithmConfig)
        self.n_agent = n_agent
        self.topics = AlgorithmConfig.topics
        from .bridge_llm import load_static_model, load_trainable_model, load_trainable_headless_model
        # 主模型
        self.main_llm_model, self.tokenizer = load_trainable_model(device=AlgorithmConfig.device_main_llm)
        self.save_model(out_dir=f'RESULT/{GlobalConfig.note}/llm_model')
        # 高温模型
        self.launcher_llm_model, self.tokenizer = load_static_model(device=AlgorithmConfig.device_launcher_llm)
        # # 奖励模型 （效果太差，使用更强的gpt-3.5模型）
        # self.reward_llm_model, self.tokenizer = self.launcher_llm_model, self.tokenizer # load_static_model(device=AlgorithmConfig.device_reward_llm)

        self.critic = ChatGLMCritic(device=AlgorithmConfig.device_main_llm)
        optimize_params = self.main_llm_model.training_para_list + list(self.critic.parameters())
        self.optimizer = Adam(optimize_params, lr=1e-4, eps=1e-3)
        self.logic_processor = Logit2Act()

        self.mcv = mcv

    
    def main_llm_answer_question(self, topic_launcher_questions):
        def pad(arr):
            pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")
            len_list = [len(token_qa_prompt) for i, token_qa_prompt in enumerate(arr)]
            max_len = max(len_list)
            for i, token_qa_prompt in enumerate(arr):
                arr[i] = [pad_id] * (max_len - len_list[i]) + token_qa_prompt
            return np.array(arr)
        token_q_prompt_array = []
        for q in topic_launcher_questions:
            token_q_prompt, gen_len, prompt = tokenize_qa(self.tokenizer, query=q, history=[])
            token_q_prompt = token_q_prompt['input_ids'].tolist()[0]
            token_q_prompt_array.append(token_q_prompt)
        token_q_prompt_array = pad(token_q_prompt_array)

        input_ids = _2tensor(token_q_prompt_array).to(AlgorithmConfig.device_main_llm)
        tokenlen_q_pad = question_token_len = input_ids.shape[-1]
        num_beams = num_return_sequences = 1
        assert num_beams >= num_return_sequences, "candidates num should greater than returns num"
        # max_new_tokens = 8
        gen_method = "greedy_search" # if num_beams == 1 else "beam_search" 
        main_model_result = self.main_llm_model.generate(input_ids=input_ids, do_sample=False, num_beams=num_beams, max_new_tokens=AlgorithmConfig.max_gen_tokens,
                            num_return_sequences=num_return_sequences, use_cache=True, num_beam_groups=1, output_scores=True,
                            output_hidden_states=False, return_dict_in_generate=True)
        main_sequences = main_model_result.sequences
        main_log_probs = get_log_prob(generated_outputs=main_model_result, input_ids=input_ids, gen_method=gen_method)
        main_gen_texts = self.tokenizer.batch_decode(main_sequences)
        main_gen_texts_only_answer = self.tokenizer.batch_decode(main_sequences[:,question_token_len:])
        for c in main_gen_texts_only_answer: print(c)
        return main_sequences, main_log_probs, main_gen_texts_only_answer, tokenlen_q_pad



    def main_llm_sft(self, n_step):
        from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

        dataset = self.read_ds(tag='sft')


        for _ in range(n_step):
            big_batch_size = len(dataset)
            self.sampler = BatchSampler(SubsetRandomSampler(range(big_batch_size)), AlgorithmConfig.sft_batch_size, drop_last=False)
            self.sampler = iter(self.sampler)
            indices = next(self.sampler)
            samples = dataset[indices]

            def pad(arr):
                pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")
                len_list = [len(token_qa_prompt) for i, token_qa_prompt in enumerate(arr)]
                max_len = max(len_list)
                for i, token_qa_prompt in enumerate(arr):
                    arr[i] = [pad_id] * (max_len - len_list[i]) + token_qa_prompt
                return np.array(arr)
            
            token_q_prompt_array = []
            for q, a in samples:
                a = a[:AlgorithmConfig.max_gen_tokens]
                token_q_prompt, gen_len, prompt = tokenize_qa(self.tokenizer, query="", history=[[q, a]])
                token_q_prompt = token_q_prompt['input_ids'].tolist()[0]
                token_q_prompt_array.append(token_q_prompt)
            token_q_prompt_array = pad(token_q_prompt_array)

            input_ids = _2tensor(token_q_prompt_array).to(AlgorithmConfig.device_main_llm)
            tokenlen_q_pad = question_token_len = input_ids.shape[-1]
            num_beams = num_return_sequences = 1
            assert num_beams >= num_return_sequences, "candidates num should greater than returns num"

            # max_new_tokens = 8
            model_inputs = self.main_llm_model.prepare_inputs_for_generation(input_ids)
            output = self.main_llm_model(**model_inputs)
            logits = output.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss = nn.functional.cross_entropy(my_view(shift_logits, [-1, 0]), my_view(shift_labels, [-1]))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            self.mcv.rec(loss.item(),'stf_loss')
            self.mcv.rec_show()


            print(loss.item())





    def interact_with_env(self, StateRecall):
        '''
            Interfacing with marl, standard method that you must implement
            (redirect to shell_env to help with history rolling)
        '''
        # from .temp import tokenize_qa
        print('\n\n\n')

        if self.ds_size(tag='sft') < AlgorithmConfig.dataset_size:
            #  获取话题问题
            with torch.no_grad(): # 此处不需要梯度，后面PPO会二次计算
                topic_launcher_questions = self.topic_launcher_propose_question()
            torch.cuda.empty_cache()
            print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-')
            # 

            
            #  主模型回答问题
            with torch.no_grad(): # 此处不需要梯度，后面PPO会二次计算
                main_sequences, main_log_probs, main_gen_texts_only_answer, tokenlen_q_pad = self.main_llm_answer_question(topic_launcher_questions)
            torch.cuda.empty_cache()
            print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-')
            # 

            if AlgorithmConfig.enable_rl:
                #  给出奖励
                with torch.no_grad(): # 此处不需要梯度，后面PPO会二次计算
                    gen_texts_answer_only, rewards = self.reward_eval_answer_question(topic_launcher_questions, main_gen_texts_only_answer)
                    rewards, masks = self.place_reward(tokenlen_q_pad, 
                                                    main_sequences, rewards, self.tokenizer.convert_tokens_to_ids("<pad>"))
                torch.cuda.empty_cache()
                print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-')

            if AlgorithmConfig.enable_stf:
                #  给出修改后的参考答案
                with torch.no_grad(): # 此处不需要梯度，后面PPO会二次计算
                    revised_answer = self.revise_answer_question(topic_launcher_questions, main_gen_texts_only_answer)
                torch.cuda.empty_cache()
                print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-')

        if AlgorithmConfig.enable_stf:
            #  训练
            self.main_llm_sft(5)
            torch.cuda.empty_cache()
            print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-')


        if AlgorithmConfig.enable_rl:
            torch.cuda.empty_cache()
            self.ppo(ppo_epochs=5, sequences=main_sequences, log_probs=main_log_probs, rewards=rewards, masks=masks, clip_param=0.2)


        action = np.zeros(shape=(1, 1))
        if StateRecall['Current-Obs-Step'][0] % AlgorithmConfig.save_step == 1: # AlgorithmConfig.save_step-1:
            self.save_model(out_dir=f'RESULT/{GlobalConfig.note}/llm_model')
        return action, StateRecall
    
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

    def save_model(self, out_dir):
        import shutil
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir,'saved')
        shutil.rmtree(model_path, ignore_errors=True)
        # shutil.copytree(AlgorithmConfig.model_path, model_path)
        self.main_llm_model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

    def place_reward(self, len_of_query, sequences, reward, pad_id):
        reward = _2tensor(reward).to(AlgorithmConfig.device_main_llm)
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
            new_log_probs = get_log_probs_with_input_ids(self.main_llm_model, sequences, log_probs.shape[1])
            with torch.no_grad():
                new_log_probs_ref = None # get_log_probs_with_input_ids(self.main_llm_model_ref, sequences.to(self.main_llm_model_ref.device), log_probs.shape[1])
                if new_log_probs_ref is not None:
                    kl_div_loss = F.kl_div(new_log_probs_ref.to(new_log_probs.device), new_log_probs, log_target=True, reduction="none").mean()
            
            # entropy = 0 # 暂时不需要熵的约束
            # compute value
            # 到奖励模型和值函数模型的输入可以是一样的都是生成的序列。
            # 生成序列同时包括state和next action
            # prepare input for critic model
            input_ids_critic = sequences.to(AlgorithmConfig.device_main_llm)
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


            loss = critic_loss + actor_loss
            if new_log_probs_ref is not None:
                loss += kl_div_loss * 0.001

            self.mcv.rec(critic_loss.item(),'critic_loss')
            self.mcv.rec(actor_loss.item(),'actor_loss')
            if new_log_probs_ref is not None:
                self.mcv.rec(kl_div_loss.item(),'kl_div_loss')
            self.mcv.rec_show()
            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("loss", loss.item())
