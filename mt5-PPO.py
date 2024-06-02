# 0. imports
import torch
from transformers import MT5Tokenizer
import os
from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer
import numpy as np
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm_notebook


# 1. load a pretrained model
# model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"my_mt5_model.pt")
checkpoint = "google/mt5-small" # 调用模型的名称
checkpoint1 = "kachipaa/Ian-model"
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(checkpoint1)
model_ref = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(checkpoint1)
tokenizer = MT5Tokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

# 2. initialize trainer
ppo_config = {"batch_size": 4}
config = PPOConfig(**ppo_config)
ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer)

# 3. encode a query
books = load_dataset("alt", "alt-parallel") # 加载数据集，数据集是json文件格式
train_dataset =books["train"]# 划分训练集测试集
test_dataset = books["test"]
LANG_TOKEN_MAPPING ={"zh":"","lo":""} # 为需要的语言建立dict
special_tokens_dict = {'additional_special_tokens': list(LANG_TOKEN_MAPPING.values())}
tokenizer.add_special_tokens(special_tokens_dict)
max_seq_len = 100 #单句的最大长度

def encode_input_str(text, target_lang, tokenizer, seq_len,
                     lang_token_map=LANG_TOKEN_MAPPING):
    # 输入文本形式的一条语料，返回对应的id形式
    target_lang_token = lang_token_map[target_lang] # 对应的特殊token

    # Tokenize and add special tokens
    input_ids = tokenizer.encode(
        text=target_lang_token + text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=seq_len)

    return input_ids[0]


def encode_target_str(text, tokenizer, seq_len,
                      lang_token_map=LANG_TOKEN_MAPPING):
    # 同上
    token_ids = tokenizer.encode(
        text=text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=seq_len)

    return token_ids[0]


def format_translation_data(translations, lang_token_map,
                            tokenizer, seq_len=128):
    # 输入文本形式（语言名和对应语料的键值对，且只有一条），返回对应语言的id形式向量
    # Choose a random 2 languages for in i/o
    langs = list(lang_token_map.keys()) # 数据集中需要使用的两种语言
    input_lang = langs[1]
    target_lang = langs[0]

    # Get the translations for the batch
    input_text = translations[input_lang]
    target_text = translations[target_lang] # 随机对两种语言互译，此时两个tensor分别表示对应语言的一条语料

    # 限制条件
    if input_text is None or target_text is None:
        return torch.tensor([0,11],dtype=torch.long),torch.tensor([0,11],dtype=torch.long)

    # 向上调用函数，将文字转为id形式
    input_token_ids = encode_input_str(
        input_text, target_lang, tokenizer, seq_len, lang_token_map)

    target_token_ids = encode_target_str(
        target_text, tokenizer, seq_len, lang_token_map)

    return input_token_ids, target_token_ids # 返回id形式的平行语对


def transform_batch(batch, lang_token_map, tokenizer):
    # 返回一个batch数据集对应的id
    # 函数内的batch指的是train_dataset的一部分
    inputs = []
    targets = []
    for translation_set in batch['translation']:
        formatted_data = format_translation_data(
            translation_set, lang_token_map, tokenizer, max_seq_len) # 对batch数据预处理，每次处理一条

        if formatted_data is None:
            continue

        input_ids, target_ids = formatted_data
        inputs.append(input_ids.unsqueeze(0))
        targets.append(target_ids.unsqueeze(0))

    batch_input_ids = torch.cat(inputs).cuda()
    batch_target_ids = torch.cat(targets).cuda()

    return batch_input_ids, batch_target_ids


def get_data_generator(dataset, lang_token_map, tokenizer, batch_size=4):
    dataset = dataset.shuffle() # 打乱数据集顺序
    for i in range(0, len(dataset), batch_size):# 在0到数据集长度上，以batchsize为间隔
        raw_batch = dataset[i:i + batch_size] # 取batchsize 大小的数据集
        # yield 函数返回一个可迭代的generator对象，可以用for循环或next（）方法遍历生成器提取结果
        yield transform_batch(raw_batch, lang_token_map, tokenizer)

# 4. generate model response
generation_kwargs = {
    "min_length": 50,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "num_beams":5,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 100,
}

data_generator = get_data_generator(test_dataset,LANG_TOKEN_MAPPING,tokenizer,batch_size=4)
for batch_idx, (input_batch, label_batch) \
            in tqdm_notebook(enumerate(data_generator)):
    with torch.no_grad():
        response_tensor = ppo_trainer.generate([item for item in input_batch],return_prompt=False, **generation_kwargs)
        response_txt = tokenizer.decode(response_tensor[0],skip_special_tokens=True)
        print(response_txt)


# 5. define a reward for response
# (this could be any reward such as human feedback or output from another model)
    reward = [torch.tensor(1.0, device=model.pretrained_model.device),torch.tensor(1.0, device=model.pretrained_model.device),torch.tensor(1.0, device=model.pretrained_model.device),torch.tensor(1.0, device=model.pretrained_model.device)]

# 6. train model with ppo
    train_stats = ppo_trainer.step([i for i in input_batch], [j for j in label_batch], reward)
