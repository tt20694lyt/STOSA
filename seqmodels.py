import numpy as np
import torch
import torch.nn as nn
# from modules import Encoder, LayerNorm, DistSAEncoder, DistMeanSAEncoder
from modules import LayerNorm, DistSAEncoder


# class SASRecModel(nn.Module):
#     def __init__(self, args):
#         super(SASRecModel, self).__init__()
#         # print("sasrecmodel")
#         self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
#         self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
#         self.item_encoder = Encoder(args)
#         self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
#         self.dropout = nn.Dropout(args.hidden_dropout_prob)
#         self.args = args
#
#         self.criterion = nn.BCELoss(reduction='none')
#         self.apply(self.init_weights)
#
#     def add_position_embedding(self, sequence):
#         # print("sasrecmodel")
#         seq_length = sequence.size(1)
#         position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
#         position_ids = position_ids.unsqueeze(0).expand_as(sequence)
#         item_embeddings = self.item_embeddings(sequence)
#         position_embeddings = self.position_embeddings(position_ids)
#         sequence_emb = item_embeddings + position_embeddings
#         sequence_emb = self.LayerNorm(sequence_emb)
#         sequence_emb = self.dropout(sequence_emb)
#
#         return sequence_emb
#
#
#     def finetune(self, input_ids):
#         # print("finetune")
#         attention_mask = (input_ids > 0).long()
#         extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
#         max_len = attention_mask.size(-1)
#         attn_shape = (1, max_len, max_len)
#         subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
#         subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
#         subsequent_mask = subsequent_mask.long()
#
#         if self.args.cuda_condition:
#             subsequent_mask = subsequent_mask.cuda()
#
#         extended_attention_mask = extended_attention_mask * subsequent_mask
#         extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#         # 添加不同的图片和其他表征的embedding
#         sequence_emb = self.add_position_embedding(input_ids)
#         image_emb =self.add_position_embedding(input_ids)
#         title_emb =self.add_position_embedding(input_ids)
#         item_encoded_layers = self.item_encoder(sequence_emb,
#                                                 # 添加好几个argument
#                                                 # figure_emb
#                                                 # 图片和文本的title 原代码是只有id的
#                                                 extended_attention_mask,
#                                                 output_all_encoded_layers=True)
#
#         sequence_output, attention_scores = item_encoded_layers[-1]
#         return sequence_output, attention_scores
#
#     def init_weights(self, module):
#         """ Initialize the weights.
#         """
#         # print("init_weight")
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
#         elif isinstance(module, LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()

# 使用这个模型，这是模型的代码。这段代码定义了一个名为 DistSAModel 的类，它是一个神经网络模型，继承自 PyTorch 的 nn.Module
#  if args.model_name == 'DistSAModel': model = DistSAModel(args=args) 这个命令选择DistSAModel模型开始训练
class DistSAModel(nn.Module):
    #  初始化
    def __init__(self, args):
        """
        嵌入层:
        item_mean_embeddings 和 item_cov_embeddings 是嵌入层，用于将项目ID转换为固定大小的向量。这些嵌入可能代表项目的不同特征。
        position_mean_embeddings 和 position_cov_embeddings 是位置嵌入层，用于给序列中的每个位置添加一个独特的向量表示。
        user_margins 是一个嵌入层，用于表示用户的某些特征或偏好。
        编码器:
        item_encoder 是一个编码器，用于处理序列数据。它可能是一个自定义的编码器，用于处理特定类型的序列数据。
        其他层:
        LayerNorm 和 dropout 是标准化和正则化层，用于改善模型的训练和泛化能力。
        权重初始化:
        self.apply(self.init_weights) 应用自定义的权重初始化方法
        """
        super(DistSAModel, self).__init__()
        self.item_mean_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.item_cov_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_mean_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.position_cov_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.user_margins = nn.Embedding(args.num_users, 1)
        self.item_encoder = DistSAEncoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        self.apply(self.init_weights)

        # 新添加  image_embeddings部分
        image_embeddings = torch.load("image_embedding.pt")
        image_embeddings = torch.stack(image_embeddings)
        print("初始image_embeddings形状:", image_embeddings.shape)
        image_embeddings = image_embeddings.squeeze(1)
        image_embeddings_cpu = image_embeddings.cpu()
        # 确保 image_embeddings 是 NumPy 数组
        if not isinstance(image_embeddings, np.ndarray):
            image_embeddings = np.array(image_embeddings_cpu)
        print("去除中间的1维度后的形状:", image_embeddings.shape)

        self.image_embeddings = torch.nn.Embedding(num_embeddings=16539, embedding_dim=512)
        self.image_embeddings.weight.data.copy_(torch.from_numpy(image_embeddings))
        self.image_embeddings.weight.requires_grad = False

        # 新添加 text_embeddings部分
        text_embeddings = torch.load("text_embedding.pt")
        text_embeddings = torch.stack(text_embeddings)
        print("初始text_embeddings形状：", text_embeddings.shape)
        text_embeddings = text_embeddings.squeeze(1)
        text_embeddings_cpu = text_embeddings.cpu()
        # 确保 text_embeddings 是 NumPy 数组
        if not isinstance(text_embeddings, np.ndarray):
            text_embeddings = np.array(text_embeddings_cpu)
        print("去除中间的1维度后的形状:", text_embeddings.shape)

        self.text_embeddings = torch.nn.Embedding(num_embeddings=16539, embedding_dim=512)
        self.text_embeddings.weight.data.copy_(torch.from_numpy(text_embeddings))
        self.text_embeddings.weight.requires_grad = False

    """
    add_position_mean_embedding
    平均位置嵌入（Mean Embedding）
    目的: 平均位置嵌入通常用于捕捉序列中每个元素的位置信息。在序列处理任务中，了解元素的位置是非常重要的，因为位置可以影响元素的含义。
    应用: 例如，在文本处理中，单词在句子中的位置会影响其语义；在时间序列分析中，数据点的时间位置对理解整个序列至关重要
    """
    def add_position_mean_embedding(self, sequence):
        # print("add_position_mean_embedding")
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)

        item_embeddings = self.item_mean_embeddings(sequence)

        # 新添加image_embeddings部分
        image_embeddings = self.image_embeddings(sequence)
        print("image_embeddings的形状：", image_embeddings.shape)  # torch.Size([100, 50, 512])
        print("item_embeddings的形状：", item_embeddings.shape)  # torch.Size([100, 50, 64])
        text_embeddings = self.text_embeddings(sequence)
        print("text_embeddings形状：", text_embeddings.shape)  # torch.Size([100, 50, 64])

        position_embeddings = self.position_mean_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        elu_act = torch.nn.ELU()
        sequence_emb = elu_act(sequence_emb)

        return sequence_emb

    """
    add_position_cov_embedding 方法
    协方差位置嵌入（Covariance Embedding）
    目的: 协方差位置嵌入可能用于捕捉序列中元素之间的关系或依赖性。协方差是衡量两个变量如何一起变化的统计量，这在序列数据中可以被用来表示元素之间的相互作用或关联。
    应用: 在某些任务中，了解序列中不同元素之间的关系是很重要的。例如，在推荐系统中，用户对不同项目的偏好可能相互关联；在语言模型中，单词之间的依赖性对于理解句子结构至关重要。
    """

    """
    结合使用的优势
    丰富的上下文信息: 通过结合使用平均位置嵌入和协方差位置嵌入，模型能够同时考虑到元素的绝对位置（通过平均位置嵌入）和元素之间的相对关系（通过协方差位置嵌入）。
    提高模型性能: 这种多维度的上下文信息可以帮助模型更好地理解和处理序列数据，从而可能提高模型在特定任务上的性能。
    """
    def add_position_cov_embedding(self, sequence):
        # print("add_position_cov_embedding")
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_cov_embeddings(sequence)
        position_embeddings = self.position_cov_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        elu_act = torch.nn.ELU()
        sequence_emb = elu_act(self.dropout(sequence_emb)) + 1

        return sequence_emb

    """
    finetune 方法
    这个方法是模型的主要部分，用于微调。
    它首先创建一个注意力掩码，用于处理序列中的填充部分。接着，它计算序列的平均嵌入和协方差嵌入。使用 item_encoder 对嵌入后的序列进行编码。最后，它返回编码后的序列输出、注意力分数和用户的边际值。
    """
    def finetune(self, input_ids, user_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * (-2 ** 32 + 1)

        mean_sequence_emb = self.add_position_mean_embedding(input_ids)
        cov_sequence_emb = self.add_position_cov_embedding(input_ids)

        item_encoded_layers = self.item_encoder(mean_sequence_emb,
                                                cov_sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)

        mean_sequence_output, cov_sequence_output, att_scores = item_encoded_layers[-1]

        margins = self.user_margins(user_ids)
        return mean_sequence_output, cov_sequence_output, att_scores, margins

    """
    init_weights 方法
    这个方法用于初始化模型中的权重。
    对于线性层和嵌入层，它使用正态分布初始化权重。对于层标准化层，它初始化偏置为0，权重为1。如果线性层有偏置，则将其初始化为0。
    """
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            module.weight.data.normal_(mean=0.01, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class DistMeanSAModel(DistSAModel):
    def __init__(self, args):
        super(DistMeanSAModel, self).__init__(args)
        self.item_encoder = DistMeanSAEncoder(args)

