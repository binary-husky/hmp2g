class LLM_Trainable_Config(object):
    output_dir = "output/firefly-qwen-7b"
    model_name_or_path = "/home/hmp/llms/Qwen-7B"
    train_file = "./data/dummy_data.jsonl"
    num_train_epochs = 1
    per_device_train_batch_size = 8
    gradient_accumulation_steps = 2
    learning_rate = 2e-4
    max_seq_length = 1024
    logging_steps = 300
    save_steps = 500
    save_total_limit = 1
    lr_scheduler_type = "constant_with_warmup"
    warmup_steps = 3000
    lora_rank = 64
    lora_alpha = 16
    lora_dropout = 0.05

    gradient_checkpointing = True
    disable_tqdm = False
    optim = "paged_adamw_32bit"
    seed = 42
    fp16 = True
    report_to = "tensorboard"
    dataloader_num_workers = 0
    save_strategy = "steps"
    weight_decay = 0
    max_grad_norm = 0.3
    remove_unused_columns = False

    ddp_find_unused_parameters = False


from .component.argument import QLoRAArguments
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModelForCausalLM
)
import bitsandbytes as bnb
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from collections import defaultdict
from component.collator import SFTDataCollator
from component.dataset import DynamicSFTDataset
from component.argument import QLoRAArguments
from component.trainer import LoRATrainer
from component.loss import TargetLMLoss


def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def verify_model_dtype(model):
    """
    查看模型种各种类型的参数的情况
    """
    dtype2param_num = defaultdict(int)  # 每种数据类型的参数量
    dtype2param_name = defaultdict(list)  # 每种数据类型的参数名称
    dtype2trainable_param_num = defaultdict(int)  # 每种数据类型参与训练的参数量
    dtype2trainable_param_name = defaultdict(list)  # 每种数据类型参与训练的参数名称
    for name, p in model.named_parameters():
        dtype = p.dtype
        dtype2param_num[dtype] += p.numel()
        dtype2param_name[dtype].append(name)
        if p.requires_grad:
            dtype2trainable_param_num[dtype] += p.numel()
            dtype2trainable_param_name[dtype].append(name)
    # 统计全部参数中，各种类型参数分布
    total = 0
    print('verify all params of the model')
    for k, v in dtype2param_num.items():
        total += v
    for k, v in dtype2param_num.items():
        print(k, v, v / total)
    for k, v in dtype2trainable_param_name.items():
        print(k, v)

    print()
    # 统计可训练参数中，各种类型参数分布
    print('verify trainable params the model')
    total_trainable = 0
    for k, v in dtype2trainable_param_num.items():
        total_trainable += v
    for k, v in dtype2trainable_param_num.items():
        print(k, v, v / total_trainable)
    for k, v in dtype2trainable_param_num.items():
        print(k, v)

class LLM_Operator():

    def __init__(self) -> None:
        parser = HfArgumentParser((QLoRAArguments, TrainingArguments))
        args, training_args = parser.parse_dict({k:v for k,v in LLM_Trainable_Config.__dict__.items() if not k.startswith('__')})
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            ),
        )
        # 加载tokenzier
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            # llama不支持fast
            use_fast=False if model.config.model_type == 'llama' else True
        )
        # QWenTokenizer比较特殊，pad_token_id、bos_token_id、eos_token_id均为None。eod_id对应的token为<|endoftext|>
        if tokenizer.__class__.__name__ == 'QWenTokenizer':
            tokenizer.pad_token_id = tokenizer.eod_id
            tokenizer.bos_token_id = tokenizer.eod_id
            tokenizer.eos_token_id = tokenizer.eod_id
        # ChatGLMTokenizer不需要设置，仅设置其他tokenizer
        elif tokenizer.__class__.__name__ != 'ChatGLMTokenizer':
            assert tokenizer.eos_token_id is not None
            assert tokenizer.bos_token_id is not None
            tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
        print(f'memory footprint of model: {model.get_memory_footprint()/(1024*1024*1024)} GB')
        # 找到所有需要插入adapter的全连接层
        target_modules = find_all_linear_names(model)
        # 初始化lora配置
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        model.config.torch_dtype = torch.float32

        # 查看模型种各种类型的参数的情况
        verify_model_dtype(model)

        # 初始化损失函数
        loss_func = TargetLMLoss(ignore_index=-100)

        # 指加载训练集
        train_dataset = DynamicSFTDataset(args.train_file, tokenizer, args.max_seq_length)
        data_collator = SFTDataCollator(tokenizer, args.max_seq_length)
        # 初始化Trainer
        trainer = LoRATrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            # tokenizer=tokenizer,
            data_collator=data_collator,
            compute_loss=loss_func
        )

    def update_sft_dataset(self, new_qa):
        pass

    def update_rl_dataset(self, new_qa):
        pass

    def run_sft_step(self):
        pass

    def run_rl_step(self):
        pass

