import os
import warnings
from dataclasses import dataclass
import wandb
import torch
from datasets import load_dataset,load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer,PreTrainedTokenizerBase
import json,random


from trl import (
    ModelConfig,
    ScriptArguments
)

from ppo_utils.ppo_config_medo1 import PPOConfig
from ppo_utils.ppo_trainer_medo1 import PPOTrainer


os.environ["WANDB_MODE"] = "offline"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'



from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser
)

class ppo_dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length = 1000,debug = 0):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
    
        newdata = []
        for da in self.data:
            if len(da['Open-ended Verifiable Question']) > 0 and len(da['Ground-True Answer']) > 0:
                newdata.append({'question':da['Open-ended Verifiable Question'],'answer':da['Ground-True Answer']})
        print(len(self.data),' -> ',len(newdata))
        self.data = newdata

        self.debug = debug     

    def __getitem__(self, index):
        return self.data[index]

    def get_prompt(self,da):
        message = [{"role": "user", "content": da['question']}]
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        input_token = self.tokenizer(
            prompt,
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )

        da['input_ids'] = input_token["input_ids"]
        return da

    def collate_fn(self, batch):
        data = [ self.get_prompt(da) for da in batch]
        input_ids = [item["input_ids"] for item in data]
        question = [item["question"] for item in data]
        answer = [item["answer"] for item in data]

        max_len = max(len(x) for x in input_ids)
        max_len = min(max_len,self.max_length)
        input_ids = [ [self.tokenizer.pad_token_id]*(max_len-len(item)) + item[:max_len] for item in input_ids]

        if self.debug > 0:
            print('[input_ids]',self.tokenizer.decode(input_ids[-1]))
            print('[question]',question[-1])
            print('[answer]',answer[-1])
            self.debug -= 1
        return {
                "input_ids": torch.LongTensor(input_ids),
                "question": question,
                "answer": answer
            }

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    output_dir = training_args.output_dir
    run_name = training_args.run_name
    if run_name not in output_dir:
        output_dir = os.path.join(output_dir,run_name)
        training_args.output_dir = output_dir
    
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, attn_implementation="flash_attention_2",num_labels=2
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.value_model_path, trust_remote_code=model_config.trust_remote_code, attn_implementation="flash_attention_2",num_labels=1
    )

    ref_policy = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path,attn_implementation="flash_attention_2")
    policy = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path,attn_implementation="flash_attention_2")

    reward_tokenizer = AutoTokenizer.from_pretrained(training_args.reward_model_path)

    if '<|eot_id|>' in tokenizer.vocab:
        assert '<|end_of_text|>' in tokenizer.vocab
        tokenizer.pad_token = '<|end_of_text|>'
        tokenizer.pad_token_id = tokenizer.encode('<|end_of_text|>',add_special_tokens=False)[0]
    assert tokenizer.pad_token_id != tokenizer.eos_token_id

    training_args.stop_token_id = tokenizer.eos_token_id

    eval_ratio = 0.1
    eval_max_num = 200
    with open(script_args.dataset_name) as f:
        data = json.load(f)
    random.shuffle(data)
    eval_num = min(int(len(data) * eval_ratio),eval_max_num)
    train_dataset = ppo_dataset(data[eval_num:],tokenizer, debug = 1)
    eval_dataset = ppo_dataset(data[:eval_num],tokenizer)

    trainer = PPOTrainer(
        config=training_args,
        processing_class=tokenizer,
        reward_processing_class = reward_tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator = train_dataset.collate_fn
    )
    trainer.train()