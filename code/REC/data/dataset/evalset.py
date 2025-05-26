# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

import torch
from torch.utils.data import Dataset
import numpy as np
import datetime
import pytz


class SeqEvalDataset(Dataset):
    def __init__(self, config, dataload, phase='valid'): # dataload来自dataload.py中的build()函数
        self.dataload = dataload
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH_TEST'] if config['MAX_ITEM_LIST_LENGTH_TEST'] else config['MAX_ITEM_LIST_LENGTH']
        self.user_seq = list(dataload.user_seq.values()) # 每个user的交互item序列
        self.time_seq = list(dataload.time_seq.values()) # 每个user的交互时间序列
        self.use_time = config['use_time']
        self.phase = phase
        self.length = len(self.user_seq)
        self.item_num = dataload.item_num

    def __len__(self):
        return self.length

    def _padding_sequence(self, sequence, max_length): # 前面填0，取最后max_length个
        sequence = list(sequence)
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]
        return sequence

    def _padding_time_sequence(self, sequence, max_length):
        sequence = list(sequence)
        # 计算需要填充的长度
        pad_len = max_length - len(sequence)
        # 在序列前面填充指定数量的0
        sequence = [0] * pad_len + sequence
        # 截取序列以确保其长度不超过max_length
        sequence = sequence[-max_length:]
        vq_time = []
        for time in sequence:
            # 将时间戳转换为UTC时区的datetime对象
            dt = datetime.datetime.fromtimestamp(time, pytz.timezone('UTC'))
            vq_time.append([dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second])
        return vq_time

    def __getitem__(self, index):
        # index代表数据集中的用户索引，范围从0到len(self.user_seq)-1
        ###
        last_num = 2 if self.phase == 'valid' else 1 
        ###

        history_seq = self.user_seq[index][:-last_num] # 取出每个user的前history_length个item
        item_seq = self._padding_sequence(history_seq, self.max_item_list_length) # 前面填0，取最后max_length个
        item_target = self.user_seq[index][-last_num] # 取出每个user的倒数第2/1个item作为target
        if self.use_time:
            history_time_seq = self.time_seq[index][:-last_num]
        else:
            history_time_seq = []
        time_seq = self._padding_time_sequence(history_time_seq, self.max_item_list_length)

        return torch.tensor(history_seq), item_seq, item_target, time_seq  # , item_length
