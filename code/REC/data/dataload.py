# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

import copy
import pickle
import os
import yaml
from collections import Counter
from logging import getLogger

import numpy as np
import pandas as pd
import torch

from REC.utils import set_color
from REC.utils.enum_type import InputType
from torch_geometric.utils import degree


class Data: # 在run.py中的dataload = load_data(config)处调用
    def __init__(self, config):
        # 初始化数据集类，加载配置并准备从头开始加载数据
        self.config = config
        self.dataset_path = config['data_path']
        self.dataset_name = config['dataset']
        self.data_split = config['data_split']
        self.item_data = config['item_data']
        self.logger = getLogger()
        self._from_scratch()

    def _from_scratch(self):
        # 从头开始加载和处理数据
        self.logger.info(set_color(f'Loading {self.__class__} from scratch with {self.data_split = }.', 'green'))
        self._load_inter_feat(self.dataset_name, self.dataset_path, self.item_data)
        self._data_processing()

    def _load_inter_feat(self, token, dataset_path, item_data=None):
        # 加载交互特征数据，如果配置了item_data，则同时加载物品特征数据
        inter_feat_path = os.path.join(dataset_path, f'{token}.csv')
        if not os.path.isfile(inter_feat_path):
            raise ValueError(f'File {inter_feat_path} not exist.')

        df = pd.read_csv(
            inter_feat_path, delimiter=',', dtype={'item_id': str, 'user_id': str, 'timestamp': int}, header=0, names=['item_id', 'user_id', 'timestamp']
        )
        self.logger.info(f'Interaction feature loaded successfully from [{inter_feat_path}].')
        self.inter_feat = df # 用户交互数据

        if item_data:
            item_data_path = os.path.join(dataset_path, f'{item_data}.csv')
            item_df = pd.read_csv(
                item_data_path, delimiter=',', dtype={'item_id': str, 'user_id': str, 'timestamp': int}, header=0, names=['item_id', 'user_id', 'timestamp']
            )
            self.item_feat = item_df # 商品数据
            self.logger.info(f'Item feature loaded successfully from [{item_data}].')

    def _data_processing(self):
        # 处理数据，包括生成ID到token的映射，处理交互特征等
        self.id2token = {}
        self.token2id = {}
        remap_list = ['user_id', 'item_id']
        for feature in remap_list:
            if feature == 'item_id' and self.item_data:
                feats = self.item_feat[feature]
                feats_raw = self.inter_feat[feature]
            else:
                feats = self.inter_feat[feature]
            new_ids_list, mp = pd.factorize(feats) # 将feats中的原始值映射为新的id，mp为所有unique的feat
            mp = ['[PAD]'] + list(mp) # [PAD]用于填充
            token_id = {t: i for i, t in enumerate(mp)} # 构建 token_id 字典，将 token 映射到对应的 ID。
            if feature == 'item_id' and self.item_data: # 处理物品特征的额外映射
                _, raw_mp = pd.factorize(feats_raw)
                for x in raw_mp:
                    if x not in token_id:
                        token_id[x] = len(token_id)
                        mp.append(x)
            mp = np.array(mp)

            self.id2token[feature] = mp
            self.token2id[feature] = token_id
            self.inter_feat[feature] = self.inter_feat[feature].map(token_id) # 将交互特征中的原始值替换为对应的 ID

        self.user_num = len(self.id2token['user_id'])
        self.item_num = len(self.id2token['item_id'])
        self.logger.info(f"{self.user_num = } {self.item_num = }")
        self.logger.info(f"{self.inter_feat['item_id'].isna().any() = } {self.inter_feat['user_id'].isna().any() = }")
        self.inter_num = len(self.inter_feat)
        self.uid_field = 'user_id'
        self.iid_field = 'item_id'
        self.user_seq = None
        self.train_feat = None
        self.feat_name_list = ['inter_feat']  # self.inter_feat

    def build(self):
        # 构建数据加载器，对交互数据进行排序并生成用户序列
        self.logger.info(f"build {self.dataset_name} dataload")
        self.sort(by='timestamp')
        user_list = self.inter_feat['user_id'].values
        item_list = self.inter_feat['item_id'].values
        timestamp_list = self.inter_feat['timestamp'].values
        grouped_index = self._grouped_index(user_list) 

        user_seq = {}
        time_seq = {}
        for uid, index in grouped_index.items():
            user_seq[uid] = item_list[index]
            time_seq[uid] = timestamp_list[index]

        # e.g.
        # user_seq = {
        #     'user1': np.array([101, 103]),
        #     'user2': np.array([102, 105]),
        #     'user3': np.array([104])
        # }
        # time_seq = {
        #     'user1': np.array([1620000000, 1620000200]),
        #     'user2': np.array([1620000100, 1620000400]),
        #     'user3': np.array([1620000300])
        # }

        self.user_seq = user_seq
        self.time_seq = time_seq
        train_feat = dict()
        indices = []

        for index in grouped_index.values():
            indices.extend(list(index)[:-2]) # 从每个用户的交互记录中剔除最后两个交互，然后将剩下的交互记录的索引收集起来
        for k in self.inter_feat:
            train_feat[k] = self.inter_feat[k].values[indices]

        if self.config['MODEL_INPUT_TYPE'] == InputType.AUGSEQ:
            train_feat = self._build_aug_seq(train_feat)
        elif self.config['MODEL_INPUT_TYPE'] == InputType.SEQ: # HLLM是这个
            train_feat = self._build_seq(train_feat)

        self.train_feat = train_feat

    def _grouped_index(self, group_by_list):
        # 根据用户ID对交互记录进行分组
        # 返回一个字典 index，其中：
        # 键：用户 ID（或其他分组键）。
        # 值：该用户的交互记录在 group_by_list 中的索引列表。
        # {
        #     'user1': [0, 2],  # 'user1' 的交互记录在索引 0 和 2
        #     'user2': [1, 4],  # 'user2' 的交互记录在索引 1 和 4
        #     'user3': [3]      # 'user3' 的交互记录在索引 3
        # }
        index = {}
        for i, key in enumerate(group_by_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)
        return index

    def _build_seq(self, train_feat):
        # 构建用户交互序列
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']+1

        uid_list, item_list_index = [], []
        seq_start = 0
        save = False
        user_list = train_feat['user_id']
        user_list = np.append(user_list, -1) # 用户 ID 列表，添加一个哨兵值 -1，用于处理最后一个用户的交互记录。
        last_uid = user_list[0]
        for i, uid in enumerate(user_list):
            if last_uid != uid:
                save = True # 切换到新用户
            if save:
                if (self.data_split is None or self.data_split == True) and i - seq_start > max_item_list_len: #如果当前用户交互记录长度超过max_item_list_len，则切分成多个序列
                    offset = (i - seq_start) % max_item_list_len
                    seq_start += offset
                    x = torch.arange(seq_start, i)
                    sx = torch.split(x, max_item_list_len)
                    for sub in sx:
                        uid_list.append(last_uid)
                        item_list_index.append(slice(sub[0], sub[-1]+1)) # 创建一个切片对象，表示从 sub[0] 到 sub[-1]（包含）的范围。
                else:
                    uid_list.append(last_uid)
                    item_list_index.append(slice(seq_start, i))  # maybe too long but will be truncated in dataloader

                save = False
                last_uid = uid
                seq_start = i

        seq_train_feat = {}
        seq_train_feat['user_id'] = np.array(uid_list)
        seq_train_feat['item_seq'] = []
        seq_train_feat['time_seq'] = []
        for index in item_list_index:
            seq_train_feat['item_seq'].append(train_feat['item_id'][index])
            seq_train_feat['time_seq'].append(train_feat['timestamp'][index])

        # seq_train_feat = {
        #     'user_id': [1, 1, 2, 3],
        #     'item_seq': [
        #         [101, 102],
        #         [103],
        #         [201, 202],
        #         [301]
        #     ],
        #     'time_seq': [
        #         [1001, 1002],
        #         [1003],
        #         [2001, 2002],
        #         [3001]
        #     ]
        # }
        return seq_train_feat

    def _build_aug_seq(self, train_feat):
        # 构建增强的用户交互序列
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']+1

        uid_list, item_list_index = [], []
        seq_start = 0
        save = False
        user_list = train_feat['user_id']
        user_list = np.append(user_list, -1)
        last_uid = user_list[0]
        for i, uid in enumerate(user_list):
            if last_uid != uid:
                save = True
            if save:
                if i - seq_start > max_item_list_len:
                    offset = (i - seq_start) % max_item_list_len
                    seq_start += offset
                    x = torch.arange(seq_start, i)
                    sx = torch.split(x, max_item_list_len)
                    for sub in sx:
                        uid_list.append(last_uid)
                        item_list_index.append(slice(sub[0], sub[-1]+1))
                else:
                    uid_list.append(last_uid)
                    item_list_index.append(slice(seq_start, i))
                save = False
                last_uid = uid
                seq_start = i

        seq_train_feat = {}
        aug_uid_list = []
        aug_item_list = []
        for uid, item_index in zip(uid_list, item_list_index):
            st = item_index.start
            ed = item_index.stop
            lens = ed - st
            for sub_idx in range(1, lens):
                aug_item_list.append(train_feat['item_id'][slice(st, st+sub_idx+1)])
                aug_uid_list.append(uid)

        seq_train_feat['user_id'] = np.array(aug_uid_list)
        seq_train_feat['item_seq'] = aug_item_list

        return seq_train_feat

    def sort(self, by, ascending=True):
        # 对交互特征数据按照指定字段排序
        if isinstance(self.inter_feat, pd.DataFrame):
            self.inter_feat.sort_values(by=by, ascending=ascending, inplace=True)

        else:
            if isinstance(by, str):
                by = [by]

            if isinstance(ascending, bool):
                ascending = [ascending]

            if len(by) != len(ascending):
                if len(ascending) == 1:
                    ascending = ascending * len(by)
                else:
                    raise ValueError(f'by [{by}] and ascending [{ascending}] should have same length.')
            for b, a in zip(by[::-1], ascending[::-1]):
                index = np.argsort(self.inter_feat[b], kind='stable')
                if not a:
                    index = index[::-1]
                for k in self.inter_feat:
                    self.inter_feat[k] = self.inter_feat[k][index]

    @property
    def avg_actions_of_users(self):
        """Get the average number of users' interaction records.

        Returns:
            numpy.float64: Average number of users' interaction records.
        """
        if isinstance(self.inter_feat, pd.DataFrame):
            return np.mean(self.inter_feat.groupby(self.uid_field).size())
        else:
            return np.mean(list(Counter(self.inter_feat[self.uid_field]).values()))

    @property
    def avg_actions_of_items(self):
        """Get the average number of items' interaction records.

        Returns:
            numpy.float64: Average number of items' interaction records.
        """
        if isinstance(self.inter_feat, pd.DataFrame):
            return np.mean(self.inter_feat.groupby(self.iid_field).size())
        else:
            return np.mean(list(Counter(self.inter_feat[self.iid_field]).values()))

    @property
    def sparsity(self):
        """Get the sparsity of this dataset.

        Returns:
            float: Sparsity of this dataset.
        """
        return 1 - self.inter_num / self.user_num / self.item_num

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        # 返回数据集的基本信息字符串
        info = [set_color(self.dataset_name, 'pink')]
        if self.uid_field:
            info.extend([
                set_color('The number of users', 'blue') + f': {self.user_num}',
                set_color('Average actions of users', 'blue') + f': {self.avg_actions_of_users}'
            ])
        if self.iid_field:
            info.extend([
                set_color('The number of items', 'blue') + f': {self.item_num}',
                set_color('Average actions of items', 'blue') + f': {self.avg_actions_of_items}'
            ])
        info.append(set_color('The number of inters', 'blue') + f': {self.inter_num}')
        if self.uid_field and self.iid_field:
            info.append(set_color('The sparsity of the dataset', 'blue') + f': {self.sparsity * 100}%')

        return '\n'.join(info)

    def copy(self, new_inter_feat):
        """Given a new interaction feature, return a new :class:`Dataset` object,
        whose interaction feature is updated with ``new_inter_feat``, and all the other attributes the same.

        Args:
            new_inter_feat (Interaction): The new interaction feature need to be updated.

        Returns:
            :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
        """
        nxt = copy.copy(self)
        nxt.inter_feat = new_inter_feat
        return nxt

    def counter(self, field):
        # 计算指定字段的出现次数
        if isinstance(self.inter_feat, pd.DataFrame):
            return Counter(self.inter_feat[field].values)
        else:
            return Counter(self.inter_feat[field])

    @property
    def user_counter(self):
        # 返回用户ID的计数器
        return self.counter('user_id')

    @property
    def item_counter(self):
        # 返回物品ID的计数器
        return self.counter('item_id')

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            The normalized interaction matrix in Tensor.
        """
        row = torch.tensor(self.train_feat[self.uid_field])
        col = torch.tensor(self.train_feat[self.iid_field]) + self.user_num
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)

        deg = degree(edge_index[0], self.user_num + self.item_num)

        norm_deg = 1. / torch.sqrt(torch.where(deg == 0, torch.ones([1]), deg))
        edge_weight = norm_deg[edge_index[0]] * norm_deg[edge_index[1]]

        return edge_index, edge_weight