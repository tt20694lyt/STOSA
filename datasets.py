import random

import torch
from torch.utils.data import Dataset

from utils import neg_sample

class PretrainDataset(Dataset):

    def __init__(self, args, user_seq, long_sequence):
        self.args = args
        self.user_seq = user_seq
        self.long_sequence = long_sequence
        self.max_len = args.max_seq_length   # main中手动指定的参数
        self.part_sequence = []
        self.split_sequence()

    def split_sequence(self):
        for seq in self.user_seq:
            input_ids = seq[-(self.max_len+2):-2] # keeping same as train set
            # print(input_ids)
            for i in range(len(input_ids)):
                self.part_sequence.append(input_ids[:i+1])

    def __len__(self):
        return len(self.part_sequence)

    def __getitem__(self, index):
        #Mask部分：隐藏或遮蔽输入句子的某些部分，然后训练模型来预测这些被遮蔽的部分
        sequence = self.part_sequence[index] # pos_items
        # sample neg item for every masked item
        masked_item_sequence = []
        neg_items = []
        # Masked Item Prediction
        item_set = set(sequence)
        for item in sequence[:-1]:
            prob = random.random()
            if prob < self.args.mask_p:
                masked_item_sequence.append(self.args.mask_id)
                neg_items.append(neg_sample(item_set, self.args.item_size))
            else:
                masked_item_sequence.append(item)
                neg_items.append(item)

        # add mask at the last position
        masked_item_sequence.append(self.args.mask_id)
        neg_items.append(neg_sample(item_set, self.args.item_size))

        # Segment Prediction
        '''Segment Prediction (or Type Embeddings in the context of BERT):
        在BERT中，为了处理成对的句子（例如，问答任务或下一个句子预测任务），输入通常由两个句子组成。
        为了区分这两个句子，BERT使用"segment embeddings"或称为"type embeddings"。
        除了这些embeddings外，还有一个"segment mask"或"segment IDs"，用于指示每个token属于哪个句子。
        例如，对于两个句子"A"和"B"，"segment IDs"可能是[0, 0, 0, ..., 1, 1, 1]，其中0表示属于句子A的token，1表示属于句子B的token。'''
        if len(sequence) < 2:
            masked_segment_sequence = sequence
            pos_segment = sequence
            neg_segment = sequence
        else:
            sample_length = random.randint(1, len(sequence) // 2)
            start_id = random.randint(0, len(sequence) - sample_length)
            neg_start_id = random.randint(0, len(self.long_sequence) - sample_length)
            pos_segment = sequence[start_id: start_id + sample_length]
            neg_segment = self.long_sequence[neg_start_id:neg_start_id + sample_length]
            masked_segment_sequence = sequence[:start_id] + [self.args.mask_id] * sample_length + sequence[
                                                                                      start_id + sample_length:]
            pos_segment = [self.args.mask_id] * start_id + pos_segment + [self.args.mask_id] * (
                        len(sequence) - (start_id + sample_length))
            neg_segment = [self.args.mask_id] * start_id + neg_segment + [self.args.mask_id] * (
                        len(sequence) - (start_id + sample_length))

        assert len(masked_segment_sequence) == len(sequence)
        assert len(pos_segment) == len(sequence)
        assert len(neg_segment) == len(sequence)

        # padding sequence
        # padding来填充较短的句子至一个固定长度
        pad_len = self.max_len - len(sequence)
        masked_item_sequence = [0] * pad_len + masked_item_sequence
        pos_items = [0] * pad_len + sequence
        neg_items = [0] * pad_len + neg_items
        masked_segment_sequence = [0]*pad_len + masked_segment_sequence
        pos_segment = [0]*pad_len + pos_segment
        neg_segment = [0]*pad_len + neg_segment

        masked_item_sequence = masked_item_sequence[-self.max_len:]
        pos_items = pos_items[-self.max_len:]
        neg_items = neg_items[-self.max_len:]

        masked_segment_sequence = masked_segment_sequence[-self.max_len:]
        pos_segment = pos_segment[-self.max_len:]
        neg_segment = neg_segment[-self.max_len:]

        # Associated Attribute Prediction  这可能指的是预测与给定输入相关或关联的属性。
        # Masked Attribute Prediction      "Masked Attribute Prediction"可能意味着某种形式的属性被遮蔽或隐藏，模型的任务是预测它
        attributes = []
        for item in pos_items:
            attribute = [0] * self.args.attribute_size
            try:
                now_attribute = self.args.item2attribute[str(item)]
                for a in now_attribute:
                    attribute[a] = 1
            except:
                pass
            attributes.append(attribute)


        assert len(attributes) == self.max_len
        assert len(masked_item_sequence) == self.max_len
        assert len(pos_items) == self.max_len
        assert len(neg_items) == self.max_len
        assert len(masked_segment_sequence) == self.max_len
        assert len(pos_segment) == self.max_len
        assert len(neg_segment) == self.max_len


        cur_tensors = (torch.tensor(attributes, dtype=torch.long),
                       torch.tensor(masked_item_sequence, dtype=torch.long),
                       torch.tensor(pos_items, dtype=torch.long),
                       torch.tensor(neg_items, dtype=torch.long),
                       torch.tensor(masked_segment_sequence, dtype=torch.long),
                       torch.tensor(pos_segment, dtype=torch.long),
                       torch.tensor(neg_segment, dtype=torch.long),)
        return cur_tensors

class SASRecDataset(Dataset):

    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]
        # print("什么是index:", index)
        # print("此时的user_seq",self.user_seq)
        # print("刚进去的items是", self.user_seq[index])
        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4] 这里的train 和target 对应 train 中的 input_ids 和 target_pos

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]  # 从 items 序列中获取除了最后三个元素以外的所有元素，这些是训练时的特征输入
            target_pos = items[1:-2]  # target_pos 是正样本目标，也就是 input_ids 的下一个时刻的项目ID。 target_pos 代表下一次用户可能互动的项目
            answer = [0] # no use #训练集不使用？

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        # print("input_id是：",  input_ids)
        # print("target_pos是：", target_pos)
        # print("answer是:", answer)
        target_neg = []
        seq_set = set(items)  # 将 items 转换为集合 seq_set，这可能用于快速检查元素是否存在于序列中，以便生成负样本。
        for _ in input_ids:  # 遍历 input_ids 中的每个元素。下划线 _ 是一个惯用法，用来表示循环中的变量将不会被使用
            target_neg.append(neg_sample(seq_set, self.args.item_size))  # 对每个 input_id，调用 neg_sample 函数来生成一个负样本并追加到 target_neg 列表中。这个函数可能会接收当前序列集合和项目总数，以生成一个不在当前用户交互序列中的项目ID。

        pad_len = self.max_len - len(input_ids)  # 计算填充长度，即最大序列长度 self.max_len 减去当前输入ID列表 input_ids 的长度。
        input_ids = [0] * pad_len + input_ids  # 在 input_ids 前面填充零，以保证它们的长度等于最大序列长度 self.max_len。 这里没有限制最大长度 这几行代码填充负责处理长度不足的
        target_pos = [0] * pad_len + target_pos  # 同理填充正样本
        target_neg = [0] * pad_len + target_neg  # 同理填充负样本

        input_ids = input_ids[-self.max_len:]  # 这里对最大长度进行限制。 这几行代码处理长度过长的 将 input_ids 的长度限制为 self.max_len。如果 input_ids 的长度超过了 self.max_len，会截取列表的最后 self.max_len 个元素。确保输入序列不会超过模型可以处理的最大长度，同理可得后两行代码
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len
#就是在这里
        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long), # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )
        # print("cur_tensors是：", cur_tensors)
        return cur_tensors

    def __len__(self):
        return len(self.user_seq)