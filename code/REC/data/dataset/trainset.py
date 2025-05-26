# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

from asyncio.log import logger
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import random
import datetime
import pytz
import math
import torch.distributed as dist

# 数据形式为 [[user_seq], [neg_item_seq]] , [mask]


class SEQTrainDataset(Dataset):
    def __init__(self, config, dataload):
        self.dataload = dataload
        self.config = config

        self.item_num = dataload.item_num
        self.train_seq = dataload.train_feat['item_seq'] # 即itemID

        self.length = len(self.train_seq)

        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+1
        self.device = config['device']
        self.random_sample = True if config['loss'] and config['loss'] == 'nce' else False # InfoNCE,则random sample负样本
        self.num_negatives = config['num_negatives']
        if self.num_negatives:
            self.num_negatives = math.ceil(self.num_negatives / dist.get_world_size() / config['train_batch_size'])
        logger.info(f"Use random sample {self.random_sample} for mask id")

    def __len__(self):
        return self.length

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item # 负样本采样：随机选择一个不在item_set中的物品

    def _padding_sequence(self, sequence, max_length, random_sample=False):
        pad_len = max_length - len(sequence)
        # 如果 random_sample 为 True，则用随机选择的负样本填充序列；否则，用 0 填充序列
        if random_sample:
            pad_seq = [self._neg_sample(sequence) for _ in range(pad_len)]
            sequence = pad_seq + sequence
        else:
            sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:] # 截取最后max_length个元素
        return torch.tensor(sequence, dtype=torch.long)

    def reconstruct_train_data(self, item_seq):
        masked_index = []
        neg_item = []
        item_seq_len = len(item_seq)
        for i in range(item_seq_len - 1):
            neg_item.append(self._neg_sample(item_seq))
            masked_index.append(1)

        item_seq = self._padding_sequence(list(item_seq), self.max_seq_length, random_sample=self.random_sample)
        if self.num_negatives:
            neg_item = []
            for _ in range(self.num_negatives):
                neg_item.append(self._neg_sample(item_seq))
        else:
            neg_item = self._padding_sequence(neg_item, self.max_seq_length, random_sample=self.random_sample)
        masked_index = self._padding_sequence(masked_index, self.max_seq_length-1)
        return torch.as_tensor(item_seq, dtype=torch.int64), torch.as_tensor(neg_item, dtype=torch.int64), torch.as_tensor(masked_index, dtype=torch.int64)

    def __getitem__(self, index):
        # 最长长度为maxlen+1, 即若max_len是5    
        # 则存在    1,2,3,4,5,6序列,
        # pos       2,3,4,5,6
        # neg       0,8,9,7,9,8
        # mask_index 1,1,1,1,1
        item_seq = self.train_seq[index]
        item_seq, neg_item, masked_index = self.reconstruct_train_data(item_seq)

        return item_seq, neg_item, masked_index

# for HLLM only
class TextSEQTrainDataset(Dataset):
    def __init__(self, config, dataload):
        self.dataload = dataload
        self.config = config

        self.item_num = dataload.item_num
        self.train_seq = dataload.train_feat['item_seq']
        self.length = len(self.train_seq)
        self.train_time_seq = dataload.train_feat['time_seq']
        self.id2token = dataload.id2token['item_id']

        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+1
        self.max_text_length = config['MAX_TEXT_LENGTH']
        self.device = config['device']

        self.text_path = config['text_path']
        self.text_keys = config['text_keys']
        self.tokenizer = AutoTokenizer.from_pretrained(config['item_pretrain_dir'], trust_remote_code=True)
        # self.pad_id = self.tokenizer.pad_token_id
        # assert self.pad_id is not None, f"pad_token_id can't be {self.pad_id}"
        self.item_prompt = config['item_prompt']
        self.item_emb_token_n = config['item_emb_token_n']
        self.num_negatives = config['num_negatives']
        self.random_sample = True if config['loss'] and config['loss'] == 'nce' else False
        if self.num_negatives:
            self.num_negatives = math.ceil(self.num_negatives / dist.get_world_size() / config['train_batch_size'])  # for llm only
        logger.info(f"Use random sample {self.random_sample} for mask id")
        logger.info(f"Text path: {self.text_path}")
        logger.info(f"Text keys: {self.text_keys}")
        logger.info(f"Item prompt: {self.item_prompt}")
        self.load_content()

    def __len__(self):
        return self.length

    def load_content(self):
        self.env = pd.read_csv(self.text_path, delimiter=',', dtype={'item_id': str})
        self.env = self.env[self.text_keys + ['item_id']]
        self.env = self.env.set_index('item_id').T.to_dict()
        logger.info(f"Text Item num: {len(self.env)}")

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length, random_sample=False): # 在loss为nce时，random sample负样本为true，否则为false
        pad_len = max_length - len(sequence)
        if random_sample:
            pad_seq = [self._neg_sample(sequence) for _ in range(pad_len)]
            sequence = pad_seq + sequence
        else:
            sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:] # 截取最后max_length个元素
        return torch.tensor(sequence, dtype=torch.long)

    def reconstruct_train_data(self, item_seq):
        masked_index = []
        neg_item = []
        item_seq_len = len(item_seq)
        for i in range(item_seq_len - 1):
            neg_item.append(self._neg_sample(item_seq))
            masked_index.append(1)

        item_seq = self._padding_sequence(list(item_seq), self.max_seq_length, random_sample=self.random_sample)
        masked_index = self._padding_sequence(masked_index, self.max_seq_length-1) # [0,...,0,1...,1]
        if self.num_negatives: # 默认为512
            neg_item = []
            for _ in range(self.num_negatives):
                neg_item.append(self._neg_sample([]))
        else:
            neg_item = self._padding_sequence(neg_item, self.max_seq_length, random_sample=self.random_sample)
        return item_seq, neg_item, masked_index

    def _padding_time_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]
        vq_time = []
        for time in sequence:
            dt = datetime.datetime.fromtimestamp(time, pytz.timezone('UTC'))
            vq_time.append([dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second])
        return torch.tensor(vq_time, dtype=torch.long)

    def __getitem__(self, index):

        item_seq = self.train_seq[index]
        item_seq, neg_item, masked_index = self.reconstruct_train_data(item_seq)
        time_seq = self.train_time_seq[index]
        time_seq = self._padding_time_sequence(list(time_seq), self.max_seq_length)
        item_seq_token = self.id2token[item_seq]
        neg_items_token = self.id2token[neg_item]
        pos_input_ids, pos_cu_input_lens, pos_position_ids = [], [], []
        neg_input_ids, neg_cu_input_lens, neg_position_ids = [], [], []

        def process_item(item):
            if item != self.id2token[0] and item not in self.env:
                # assert item in self.env, f"{item}"
                logger.info(f"{item} not in self.env")
            item_i = self.env.get(item, {})
            text_str = ""
            if len(item_i):
                text_str = f"{self.item_prompt}"
                for key in self.text_keys:
                    value = item_i[key]
                    if value and str(value) != 'nan':
                        text_str += f"{key}: {value}"

            ids = self.tokenizer.encode(text_str)
            ids = ids[:self.max_text_length]
            mask = [1] * len(ids)
            return ids, mask

        for item in item_seq_token:
            ids, _ = process_item(item)
            pos_input_ids.extend(ids + [0] * self.item_emb_token_n)
            pos_cu_input_lens.append(len(ids) + self.item_emb_token_n)
            pos_position_ids.extend((torch.arange(len(ids) + self.item_emb_token_n) + (self.max_text_length - len(ids))).tolist())

        for neg in neg_items_token:
            ids, _ = process_item(neg)
            neg_input_ids.extend(ids + [0] * self.item_emb_token_n)
            neg_cu_input_lens.append(len(ids) + self.item_emb_token_n)
            neg_position_ids.extend((torch.arange(len(ids) + self.item_emb_token_n) + (self.max_text_length - len(ids))).tolist())

        outputs = {
            "pos_item_ids": torch.as_tensor(item_seq, dtype=torch.int64),
            "neg_item_ids": torch.as_tensor(neg_item, dtype=torch.int64),
            "pos_input_ids": torch.as_tensor(pos_input_ids, dtype=torch.int64), # 正样本 item 文本的 token ID 序列，拼接在一起的（不同 item 的文本后跟 padding token）
            "pos_cu_input_lens": torch.as_tensor(pos_cu_input_lens, dtype=torch.int64), # 正样本每个文本的 token 长度（包括补的 token），用来切分 pos_input_ids
            "pos_position_ids": torch.as_tensor(pos_position_ids, dtype=torch.int64), # 	正样本的 position encoding，长度等于 pos_input_ids，用于位置嵌入。
            "neg_input_ids": torch.as_tensor(neg_input_ids, dtype=torch.int64),
            "neg_cu_input_lens": torch.as_tensor(neg_cu_input_lens, dtype=torch.int64),
            "neg_position_ids": torch.as_tensor(neg_position_ids, dtype=torch.int64),
            "attention_mask": torch.as_tensor(masked_index, dtype=torch.int64), # 用于指示哪些位置是被 mask（即需要预测）的。通常是全1（除了padding部分），类似 BERT-style mask。
            "time_ids": torch.as_tensor(time_seq, dtype=torch.int64), # 每个 item 的时间戳转换成 [年, 月, 日, 时, 分, 秒]，用于时间编码模块。
        }
        return outputs
