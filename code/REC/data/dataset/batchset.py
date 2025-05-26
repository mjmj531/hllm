# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

from torch.utils.data import Dataset

import torch
import pandas as pd
from transformers import AutoTokenizer
import logging


class BatchTextDataset(Dataset):
    # 初始化数据集，加载配置和数据
    def __init__(self, config, dataload):
        self.item_num = dataload.item_num
        self.item_list = dataload.id2token['item_id']
        self.max_text_length = config['MAX_TEXT_LENGTH']
        self.device = config['device']

        self.text_path = config['text_path']
        self.text_keys = config['text_keys']
        self.tokenizer = AutoTokenizer.from_pretrained(config['item_pretrain_dir'], trust_remote_code=True)
        # self.pad_id = self.tokenizer.pad_token_id
        # assert self.pad_id is not None, f"pad_token_id can't be {self.pad_id}"
        self.item_prompt = config['item_prompt']
        self.item_emb_token_n = config['item_emb_token_n']
        self.logger = logging.getLogger()
        self.load_content()

    def __len__(self):
        return self.item_num

    # 加载文本内容到内存中
    # 加载 CSV 文件并将内容按 item_id 分组。
#    # 通过 to_dict() 转成以 item_id 为键、对应文本字段为值的字典格式，方便快速索引。
#     self.env = {
#     '1001': {'title': 'Apple', 'brand': 'FruitCo'},
#     '1002': {'title': 'Banana', 'brand': 'Tropica'},
#     ...
# }

    def load_content(self):
        self.env = pd.read_csv(self.text_path, delimiter=',', dtype={'item_id': str})
        self.env = self.env[self.text_keys + ['item_id']]
        self.env = self.env.set_index('item_id').T.to_dict()
        self.logger.info(f"Text Item num: {len(self.env)}")

    # 获取指定索引的文本数据项
    def __getitem__(self, index):
        def process_item(item):
            if item != self.item_list[0] and item not in self.env:
                self.logger.info(f"{item} not in self.env")
            item_i = self.env.get(item, {})
            text_str = ""
            if len(item_i):
                text_str = f"{self.item_prompt}"
                for key in self.text_keys:
                    value = item_i[key]
                    if value and str(value) != 'nan':
                        text_str += f"{key}: {value}"

            ids = self.tokenizer.encode(text_str) # encode为token_id
            ids = ids[:self.max_text_length]
            mask = [1] * len(ids)
            return ids, mask

        if index == 0 or index == self.item_num:
            item_token_i = ""
        else:
            item_token_i = self.item_list[index]
        pos_input_ids, pos_cu_input_lens, pos_position_ids = [], [], []
        ids, _ = process_item(item_token_i)
        pos_input_ids.extend(ids + [0] * self.item_emb_token_n)
        pos_cu_input_lens.append(len(ids) + self.item_emb_token_n)
        pos_position_ids.extend((torch.arange(len(ids) + self.item_emb_token_n) + (self.max_text_length - len(ids))).tolist()) # 加了一项偏置
        outputs = {
            "pos_item_ids": torch.as_tensor(index, dtype=torch.int64), 
            "pos_input_ids": torch.as_tensor(pos_input_ids, dtype=torch.int64), 
            "pos_cu_input_lens": torch.as_tensor(pos_cu_input_lens, dtype=torch.int64),
            "pos_position_ids": torch.as_tensor(pos_position_ids, dtype=torch.int64)
        }
        return outputs # dict