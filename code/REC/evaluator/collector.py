# Copyright (c) 2024 westlake-repl
# SPDX-License-Identifier: MIT

from .register import Register
import torch
import copy
import numpy as np


class DataStruct(object):
    # 定义一个数据结构类，用于存储和管理数据

    def __init__(self):
        # 初始化一个空的字典来存储数据
        self._data_dict = {}

    def __getitem__(self, name: str):
        # 获取字典中指定键的值
        return self._data_dict[name]

    def __setitem__(self, name: str, value):
        # 设置字典中指定键的值
        self._data_dict[name] = value

    def __delitem__(self, name: str):
        # 删除字典中指定键的值
        self._data_dict.pop(name)

    def __contains__(self, key: str):
        # 检查字典中是否存在指定的键
        return key in self._data_dict

    def get(self, name: str):
        # 获取字典中指定键的值，如果键不存在则抛出异常
        if name not in self._data_dict:
            raise IndexError("Can not load the data without registration !")
        return self[name]

    def set(self, name: str, value):
        # 设置字典中指定键的值
        self._data_dict[name] = value

    def update_tensor(self, name: str, value: torch.Tensor):
        # 更新字典中指定键的张量值，如果键不存在则添加新的张量值
        if name not in self._data_dict:
            self._data_dict[name] = value.cpu().clone().detach()
        else:
            if not isinstance(self._data_dict[name], torch.Tensor):
                raise ValueError("{} is not a tensor.".format(name))
            self._data_dict[name] = torch.cat((self._data_dict[name], value.cpu().clone().detach()), dim=0)

    def __str__(self):
        # 返回一个字符串，显示字典中包含的所有键
        data_info = '\nContaining:\n'
        for data_key in self._data_dict.keys():
            data_info += data_key + '\n'
        return data_info

class Collector(object):
    """The collector is used to collect the resource for evaluator.
        As the evaluation metrics are various, the needed resource not only contain the recommended result
        but also other resource from data and model. They all can be collected by the collector during the training
        and evaluation process.

        This class is only used in Trainer.

    """

    def __init__(self, config):
        self.config = config
        self.data_struct = DataStruct()
        self.register = Register(config) # 一个 Register 实例，用于检查哪些资源需要被收集。
        self.full = True
        self.topk = self.config['topk']
        self.device = self.config['device']

    def data_collect(self, train_data):
        """ Collect the evaluation resource from training data.
            Args:
                train_data (AbstractDataLoader): the training dataloader which contains the training data.

        """
        if self.register.need('data.num_items'):
            item_id = 'item_id'
            self.data_struct.set('data.num_items', train_data.dataset.item_num)
        if self.register.need('data.num_users'):
            user_id = 'user_id'
            self.data_struct.set('data.num_users', train_data.dataset.user_num)
        if self.register.need('data.count_items'):
            self.data_struct.set('data.count_items', train_data.dataset.item_counter)
        if self.register.need('data.count_users'):
            self.data_struct.set('data.count_users', train_data.dataset.user_counter)

    def _average_rank(self, scores):
        # 计算某个有序张量的平均排名
        """Get the ranking of an ordered tensor, and take the average of the ranking for positions with equal values.

        Args:
            scores(tensor): an ordered tensor, with size of `(N, )`

        Returns:
            torch.Tensor: average_rank

        Example:
            >>> average_rank(tensor([[1,2,2,2,3,3,6],[2,2,2,2,4,5,5]])) # 这里输入的是每个张量的排名
            tensor([[1.0000, 3.0000, 3.0000, 3.0000, 5.5000, 5.5000, 7.0000],
            [2.5000, 2.5000, 2.5000, 2.5000, 5.0000, 6.5000, 6.5000]])

        Reference:
            https://github.com/scipy/scipy/blob/v0.17.1/scipy/stats/stats.py#L5262-L5352

        """
        length, width = scores.shape
        true_tensor = torch.full((length, 1), True, dtype=torch.bool, device=self.device)

        obs = torch.cat([true_tensor, scores[:, 1:] != scores[:, :-1]], dim=1)
        # bias added to dense
        bias = torch.arange(0, length, device=self.device).repeat(width).reshape(width, -1). \
            transpose(1, 0).reshape(-1)
        dense = obs.view(-1).cumsum(0) + bias

        # cumulative counts of each unique value
        count = torch.where(torch.cat([obs, true_tensor], dim=1))[1]
        # get average rank
        avg_rank = .5 * (count[dense] + count[dense - 1] + 1).view(length, -1)

        return avg_rank

    def eval_batch_collect(
        self, scores_tensor: torch.Tensor, positive_u: torch.Tensor, positive_i: torch.Tensor, interaction=None
    ):
        """ Collect the evaluation resource from batched eval data and batched model output.
            Args:
                scores_tensor (Torch.Tensor): the output tensor of model with the shape of `(N, )`
                interaction(Interaction): batched eval data.
                positive_u(Torch.Tensor): the row index of positive items for each user.
                positive_i(Torch.Tensor): the positive item id for each user.
        """
        if self.register.need('rec.items'):

            # get topk
            _, topk_idx = torch.topk(scores_tensor, max(self.topk), dim=-1)  # n_users x k
            self.data_struct.update_tensor('rec.items', topk_idx)

        if self.register.need('rec.topk'):

            _, topk_idx = torch.topk(scores_tensor, max(self.topk), dim=-1)  # n_users x k
            pos_matrix = torch.zeros_like(scores_tensor, dtype=torch.int) # scores_tensor.shape: n_users x n_items
            pos_matrix[positive_u, positive_i] = 1
            pos_len_list = pos_matrix.sum(dim=1, keepdim=True) # shape: n_users x 1，每个user的正样本总数
            pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx) # 检查这些物品是否是用户真正感兴趣的正样本物品
            result = torch.cat((pos_idx, pos_len_list), dim=1)
            self.data_struct.update_tensor('rec.topk', result)

        if self.register.need('rec.meanrank'):

            desc_scores, desc_index = torch.sort(scores_tensor, dim=-1, descending=True)

            # get the index of positive items in the ranking list
            pos_matrix = torch.zeros_like(scores_tensor)
            pos_matrix[positive_u, positive_i] = 1
            pos_index = torch.gather(pos_matrix, dim=1, index=desc_index)

            avg_rank = self._average_rank(desc_scores)
            pos_rank_sum = torch.where(pos_index == 1, avg_rank, torch.zeros_like(avg_rank)).sum(dim=-1, keepdim=True)

            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            user_len_list = desc_scores.argmin(dim=1, keepdim=True)
            result = torch.cat((pos_rank_sum, user_len_list, pos_len_list), dim=1)
            self.data_struct.update_tensor('rec.meanrank', result)

        if self.register.need('rec.score'):

            self.data_struct.update_tensor('rec.score', scores_tensor)

        # if self.register.need('data.label'):
            # self.label_field = self.config['LABEL_FIELD']
            # self.data_struct.update_tensor('data.label', interaction[self.label_field].to(self.device))

    def model_collect(self, model: torch.nn.Module):
        """ Collect the evaluation resource from model.
            Args:
                model (nn.Module): the trained recommendation model.
        """
        pass
        # TODO:

    def eval_collect(self, eval_pred: torch.Tensor, data_label: torch.Tensor):
        """ Collect the evaluation resource from total output and label.
            It was designed for those models that can not predict with batch.
            Args:
                eval_pred (torch.Tensor): the output score tensor of model.
                data_label (torch.Tensor): the label tensor.
        """
        if self.register.need('rec.score'):
            self.data_struct.update_tensor('rec.score', eval_pred)

        if self.register.need('data.label'):
            self.label_field = self.config['LABEL_FIELD']
            self.data_struct.update_tensor('data.label', data_label.to(self.device))

    def distributed_concat(self, tensor, num_total_examples):
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)
        # truncate the dummy elements added by SequentialDistributedSampler
        return concat[:num_total_examples]

    def get_data_struct(self):
        # 这段代码的作用是将收集到的所有评估资源以一个独立的副本形式返回，并在返回后清理掉一些特定的资源，为下一次评估做准备。
        """ Get all the evaluation resource that been collected.
            And reset some of outdated resource.
        """
        returned_struct = copy.deepcopy(self.data_struct)
        for key in ['rec.topk', 'rec.meanrank', 'rec.score', 'rec.items', 'data.label']:
            if key in self.data_struct:
                del self.data_struct[key] # 这些资源在返回后被重置，意味着在下一次评估时，这些资源将被重新收集。
        return returned_struct
