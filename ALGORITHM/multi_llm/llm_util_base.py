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

class Generator:
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        self.value = yield from self.gen

    def get_results(self):
        for i in self: pass
        return self.value

class SimpleDatasetManager():
    def read_ds(self, tag):
        with open(f'RESULT/{GlobalConfig.note}/hmp_{tag}_datasets.jsonl', 'r', encoding='utf8') as f:
            dataset = []
            for pp in f.readlines():
                d = json.loads(pp)
                dataset.append([d['topic_launcher_questions'], d['revised_text']])
        return np.array(dataset, dtype=object)

    def add_ds(self, dat, tag):
        with open(f'RESULT/{GlobalConfig.note}/hmp_{tag}_datasets.jsonl', 'a', encoding='utf8') as f:
            write_line = json.dumps(dat, ensure_ascii=False) + '\n'
            f.write(write_line)

    def ds_size(self, tag):
        os.makedirs(f'RESULT/{GlobalConfig.note}', exist_ok=True)
        if os.path.exists(f'RESULT/{GlobalConfig.note}/hmp_{tag}_datasets.jsonl'):
            with open(f'RESULT/{GlobalConfig.note}/hmp_{tag}_datasets.jsonl', 'r', encoding='utf8') as f:
                q = f.readlines()
                return len(q)
        return 0

class TopicLauncher():

    def topic_launcher_propose_question(self):
        from .foundation import AlgorithmConfig
        def pad(arr):
            pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")
            len_list = [len(token_qa_prompt) for i, token_qa_prompt in enumerate(arr)]
            max_len = max(len_list)
            for i, token_qa_prompt in enumerate(arr):
                arr[i] = [pad_id] * (max_len - len_list[i]) + token_qa_prompt
            return np.array(arr)
        token_q_prompt_array = []
        for _ in range(10):
            token_q_prompt, gen_len, prompt = tokenize_qa(self.tokenizer, 
                query=AlgorithmConfig.topic_launcher_prompt.replace('TOPIC_REPLACE', random.choice(self.topics)), history=[])
            token_q_prompt = token_q_prompt['input_ids'].tolist()[0]
            token_q_prompt_array.append(token_q_prompt)
        token_q_prompt_array = pad(token_q_prompt_array)

        num_beams = 4
        num_return_sequences = 1
        input_ids = _2tensor(token_q_prompt_array).to(AlgorithmConfig.device_launcher_llm)
        question_token_len = input_ids.shape[-1]
        model_result = self.launcher_llm_model.generate(input_ids=input_ids, do_sample=True, num_beams=num_beams, max_new_tokens=AlgorithmConfig.max_gen_tokens,
                                num_return_sequences=num_return_sequences, use_cache=True, num_beam_groups=1, output_scores=False, temperature=2.50,
                                output_hidden_states=False, return_dict_in_generate=True)
        sequences = model_result.sequences
        gen_texts = self.tokenizer.batch_decode(sequences)
        gen_texts_answer_only = self.tokenizer.batch_decode(sequences[:, question_token_len:])
        for i, c in enumerate(gen_texts_answer_only):
            gen_texts_answer_only[i] = gen_texts_answer_only[i].lstrip('"').rstrip('"').lstrip('「').rstrip('」')
            print(gen_texts_answer_only[i])
        return gen_texts_answer_only
    

class VtTopicLauncher():

    def topic_launcher_propose_question(self):
        from .foundation import AlgorithmConfig
        from void_terminal.crazy_functions.crazy_utils import request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency
        from void_terminal.request_llm.bridge_all import predict_no_ui_long_connection
        from .json_io import GptJsonIO
        default_args = vt.get_plugin_default_kwargs()
        # construct an gpt request in case that we need to repair a broken json
        gpt_req_fn = lambda x, p: predict_no_ui_long_connection(inputs=x, llm_kwargs=default_args['llm_kwargs'], history=[], sys_prompt=p)

        # -=-=-=-=- <1> create reply json schema =-=-=-=-
        from pydantic import BaseModel, Field
        class Schema(BaseModel):
            topic: str = Field(description="The topic.")
            question: str = Field(description="The question you need to raise.")
        gjio = GptJsonIO(Schema)
        formatting_sys_prompt = gjio.generate_input()

        # -=-=-=-=- <2> create reply json schema =-=-=-=-
        inputs_array = [AlgorithmConfig.topic_launcher_prompt.replace('TOPIC_REPLACE', random.choice(self.topics)) for _ in range(10)]
        results = request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency(
            inputs_array, 
            inputs_array, 
            default_args["llm_kwargs"], 
            default_args["chatbot_with_cookie"], 
            history_array=[[] for _ in range(len(inputs_array))],
            sys_prompt_array=[formatting_sys_prompt for _ in inputs_array], 
        )
        results = Generator(results).get_results()
        # =-=-=-=- digest json reply =-=-=-=-
        gen_texts_answer_only = results[1::2]
        gpt_req_fn = lambda x, p: predict_no_ui_long_connection(inputs=x, llm_kwargs=default_args['llm_kwargs'], history=[], sys_prompt=p)
        for i, c in enumerate(gen_texts_answer_only): 
            try:
                gen_texts_answer_only[i] = gjio.generate_output_auto_repair(gen_texts_answer_only[i], gpt_req_fn)
                gen_texts_answer_only[i] = gen_texts_answer_only[i].question
            except:
                gen_texts_answer_only[i] = gen_texts_answer_only[i]

        return gen_texts_answer_only
    

class RewardEvaluator():


    def reward_eval_answer_question(self, topic_launcher_questions, main_gen_texts_only_answer):
        from void_terminal.crazy_functions.crazy_utils import request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency
        from .foundation import AlgorithmConfig

        inputs_array = []
        history_array = []
        preference = '是'
        for q, a in zip(topic_launcher_questions, main_gen_texts_only_answer):
            inputs = AlgorithmConfig.reward_model_prompt.replace('Q_REPLACE', q).replace('A_REPLACE', a)
            inputs_array.append(inputs)
            history_array.append([])

        default_args = vt.get_plugin_default_kwargs()
        results = request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency(
            inputs_array, inputs_array, default_args["llm_kwargs"], 
            default_args["chatbot_with_cookie"], history_array, ["仅回答“是”或“否”" for _ in inputs_array], 
            refresh_interval=0.2, scroller_max_len=30,
            handle_token_exceed=True, show_user_at_complete=False,
        )

        gen = Generator(results)
        for i in gen: pass
        results = gen.value
        gen_texts_answer_only = results[1::2]
        for c in gen_texts_answer_only: print(c)
        rewards = []
        for c in gen_texts_answer_only: 
            if c.startswith(preference):
                rewards.append(0.0)
            else:
                rewards.append(-1.0)
        
        for q, a, g, r in zip(topic_launcher_questions, main_gen_texts_only_answer, gen_texts_answer_only, rewards):
            pp = {
                "topic_launcher_questions": q,
                "main_gen_texts_only_answer": a,
                "gen_texts_answer_only": g,
                "rewards": r,
            }
            self.add_ds(pp, tag='rl')

        rewards = np.expand_dims(np.array(rewards), -1)

        return gen_texts_answer_only, rewards
    

    # def reward_eval_answer_question_(self, topic_launcher_questions, main_gen_texts_only_answer):
    #     def pad(arr):
    #         pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")
    #         len_list = [len(token_qa_prompt) for i, token_qa_prompt in enumerate(arr)]
    #         max_len = max(len_list)
    #         for i, token_qa_prompt in enumerate(arr):
    #             arr[i] = [pad_id] * (max_len - len_list[i]) + token_qa_prompt
    #         return np.array(arr)
    #     token_q_prompt_array = []
    #     for q, a in zip(topic_launcher_questions, main_gen_texts_only_answer):
    #         token_q_prompt, gen_len, prompt = tokenize_qa(self.tokenizer, query=f"问题：{q}\n回答：{a}\n\n以上回答是否包含夹杂英文？仅回答“是”或“否”。", history=[])
    #         token_q_prompt = token_q_prompt['input_ids'].tolist()[0]
    #         token_q_prompt_array.append(token_q_prompt)
    #     token_q_prompt_array = pad(token_q_prompt_array)

    #     input_ids = _2tensor(token_q_prompt_array).to(AlgorithmConfig.device_reward_llm)
    #     question_token_len = input_ids.shape[-1]
    #     num_beams = num_return_sequences = 1
    #     model_result = self.reward_llm_model.generate(input_ids=input_ids, do_sample=True, num_beams=num_beams, max_new_tokens=AlgorithmConfig.max_gen_tokens,
    #                             num_return_sequences=num_return_sequences, use_cache=True, num_beam_groups=1, output_scores=False, temperature=0.1,
    #                             output_hidden_states=False, return_dict_in_generate=True)
    #     sequences = model_result.sequences
    #     gen_texts = self.tokenizer.batch_decode(sequences)
    #     gen_texts_answer_only = self.tokenizer.batch_decode(sequences[:, question_token_len:])
    #     rewards = []
    #     for c in gen_texts_answer_only: 
    #         print(c)
    #         if c.startswith('否'):
    #             rewards.append(1.0)
    #         else:
    #             rewards.append(0.0)
    #     rewards = np.expand_dims(np.array(rewards), -1)
    #     return gen_texts_answer_only, rewards


class Reviser():

    def revise_answer_question_CoT(self, topic_launcher_questions, main_gen_texts_only_answer):
        
        from .foundation import AlgorithmConfig
        from void_terminal.crazy_functions.crazy_utils import request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency
        from void_terminal.request_llm.bridge_all import predict_no_ui_long_connection
        from .json_io import GptJsonIO

        inputs_array = []
        history_array = []
        for q, a in zip(topic_launcher_questions, main_gen_texts_only_answer):
            inputs = f"问题：{q}\n"
            inputs_array.append(inputs)
            history_array.append([])

        default_args = vt.get_plugin_default_kwargs()

        # -=-=-=-=- create json schema =-=-=-=-
        from pydantic import BaseModel, Field
        class Schema(BaseModel):
            revised_answer: str = Field(description="the answer to the question.")
        gjio = GptJsonIO(Schema)
        formatting_sys_prompt = gjio.generate_input()
        
        # -=-=-=-=- send request - chain of thought - step 1: creating =-=-=-=-
        results_step1 = request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency(
            inputs_array, 
            inputs_array, 
            default_args["llm_kwargs"], 
            default_args["chatbot_with_cookie"], 
            history_array, 
            [AlgorithmConfig.reviser_prompt for _ in inputs_array], 
            refresh_interval=0.2, scroller_max_len=30,
            handle_token_exceed=True, show_user_at_complete=False,
        )
        gen = Generator(results_step1)
        for i in gen: pass
        results_step1 = gen.value

        # -=-=-=-=- send request - chain of thought - step 2: formatting =-=-=-=-
        history_array = [[AlgorithmConfig.reviser_prompt, '明白', q,a] for q,a in zip(results_step1[0::2], results_step1[1::2])]
        inputs_array = ['Change your answer to json format.' + formatting_sys_prompt for _ in history_array]
        results = request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency(
            inputs_array, 
            inputs_array, 
            default_args["llm_kwargs"], 
            default_args["chatbot_with_cookie"], 
            history_array, 
            ["" for _ in inputs_array], 
            refresh_interval=0.2, scroller_max_len=30,
            handle_token_exceed=True, show_user_at_complete=False,
        )
        gen = Generator(results)
        for i in gen: pass
        results = gen.value

        # -=-=-=-=- digest json reply =-=-=-=-
        gen_texts_answer_only = results[1::2]
        # construct an gpt request in case that we need to repair a broken json
        gpt_req_fn = lambda x, p: predict_no_ui_long_connection(inputs=x, llm_kwargs=vt.get_chat_default_kwargs()['llm_kwargs'], history=[], sys_prompt=p)
        for i, c in enumerate(gen_texts_answer_only): 
            try:
                gen_texts_answer_only[i] = gjio.generate_output_auto_repair(gen_texts_answer_only[i], gpt_req_fn)
                gen_texts_answer_only[i] = gen_texts_answer_only[i].revised_answer
            except:
                pass


        # -=-=-=-=- write file =-=-=-=-
        revised_text = gen_texts_answer_only

        for q, a, g in zip(topic_launcher_questions, revised_text, main_gen_texts_only_answer):
            pp = {
                "topic_launcher_questions": q,
                "revised_text": a,
                "main_text": g,
            }
            self.add_ds(pp, tag='sft')

        return gen_texts_answer_only




    def revise_answer_question(self, topic_launcher_questions, main_gen_texts_only_answer):

        from .foundation import AlgorithmConfig
        from void_terminal.crazy_functions.crazy_utils import request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency
        from void_terminal.request_llm.bridge_all import predict_no_ui_long_connection
        from .json_io import GptJsonIO
        default_args = vt.get_plugin_default_kwargs()

        inputs_array = []
        history_array = []
        for q, a in zip(topic_launcher_questions, main_gen_texts_only_answer):
            inputs = f"问题：{q}\n"
            inputs_array.append(inputs)
            history_array.append([])


        # -=-=-=-=- create json schema =-=-=-=-
        from pydantic import BaseModel, Field
        class Schema(BaseModel):
            answer: str = Field(description="The answer to the question.")
        gjio = GptJsonIO(Schema)
        formatting_sys_prompt = gjio.generate_input()

        # -=-=-=-=-=-=-=-=-
        inputs_array_ = [ii+AlgorithmConfig.reviser_prompt_input for ii in inputs_array]
        results = request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency(
            inputs_array_,
            inputs_array_,
            default_args["llm_kwargs"],
            default_args["chatbot_with_cookie"],
            history_array,
            [formatting_sys_prompt for _ in inputs_array],
            refresh_interval=0.2, scroller_max_len=30,
            handle_token_exceed=True, show_user_at_complete=False,
        )
        gen = Generator(results)
        for i in gen: pass
        results = gen.value

        # =-=-=-=- digest json reply =-=-=-=-
        gen_texts_answer_only = results[1::2]
        # construct an gpt request in case that we need to repair a broken json
        gpt_req_fn = lambda x, p: predict_no_ui_long_connection(inputs=x, llm_kwargs=default_args['llm_kwargs'], history=[], sys_prompt=p)
        for i, c in enumerate(gen_texts_answer_only): 
            try:
                gen_texts_answer_only[i] = gjio.generate_output_auto_repair(gen_texts_answer_only[i], gpt_req_fn)
                gen_texts_answer_only[i] = gen_texts_answer_only[i].answer
            except:
                pass
        # =-=-=-=- write file =-=-=-=-
        revised_text = gen_texts_answer_only

        for q, a, g in zip(topic_launcher_questions, revised_text, main_gen_texts_only_answer):
            pp = {
                "topic_launcher_questions": q,
                "revised_text": a,
                "main_text": g,
            }
            self.add_ds(pp, tag='sft')

        return gen_texts_answer_only


