import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
from opt_einsum import contract
from src.model.HAT import HATM
import numpy as np


from train_parser import generate_parser
parser = generate_parser()
args = parser.parse_args()

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.0, act="relu"):
        super().__init__()
        self.num_layers = 1
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.dropout = dropout
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(self.num_layers - 1))
        if act == "relu":
            self.act_fn = F.leaky_relu_
        elif act == "gelu":
            self.act_fn = F.gelu

    def forward(self, x):
        if not hasattr(self, 'act_fn'):
            self.act_fn = F.relu
        for i, layer in enumerate(self.layers):
            x = self.act_fn(layer(x),inplace=False) if i < self.num_layers - 1 else layer(x)
            if hasattr(self, 'dropouts') and i < self.num_layers - 1:
                x = self.dropouts[i](x)
        return x

class lableLSTM(nn.Module):
    def __init__(self):
        super(lableLSTM, self).__init__()
        self.input_size = 1024
        self.hidden = self.input_size // 2
        self.num_layers = 1
        self.lstm=nn.LSTM(self.input_size,self.hidden,self.num_layers,batch_first=True, bidirectional=True)
    def forward(self, lable_data):
        outputs, _ = self.lstm(lable_data)

        return outputs

class ModelLSTM(nn.Module):
    def __init__(self):
        super(ModelLSTM, self).__init__()
        self.input_size = 1024
        self.hidden = self.input_size // 2
        self.num_layers = 2
        self.lstm=nn.LSTM(self.input_size,self.hidden,self.num_layers,batch_first=True, bidirectional=True)
    def forward(self, lable_data):
        outputs, _ = self.lstm(lable_data)

        return outputs

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim ** -0.5
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, queries, keys, values, mask=None):

        queries = queries.transpose(0, 1)
        b, n, _, h = queries.size(0), queries.size(1), queries.size(2), self.heads
        ueries = self.query(queries)
        keys = self.key(keys)
        values = self.value(values)
        queries = ueries.view(b, n, h, -1).transpose(1, 2)
        keys = keys.view(b, args.max_length, h, -1).transpose(1, 2)
        values = values.view(b, args.max_length, h, -1).transpose(1, 2)
        dots = torch.einsum('bhid,bhjd->bhij', queries, keys) * self.scale
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'Mask has incorrect dimensions'
            mask = mask[:, None, :].expand(-1, n, -1)
            dots.masked_fill_(~mask, float('-inf'))
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, values)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return out

class CrossAttention_fusion(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim ** -0.5
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, queries, keys, values, mask=None):
        b, n, _, h = queries.size(0),queries.size(1), queries.size(2), self.heads
        ueries = self.query(queries)
        keys = self.key(keys)
        values = self.value(values)
        queries = ueries.view(b, n, h, -1).transpose(1, 2)
        keys = keys.view(b, n, h, -1).transpose(1, 2)
        values = values.view(b, n, h, -1).transpose(1, 2)
        # keys = keys.view(b, args.max_length, h, -1).transpose(1, 2)
        # values = values.view(b, args.max_length, h, -1).transpose(1, 2)
        dots = torch.einsum('bhid,bhjd->bhij', queries, keys) * self.scale
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'Mask has incorrect dimensions'
            mask = mask[:, None, :].expand(-1, n, -1)
            dots.masked_fill_(~mask, float('-inf'))
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, values)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return out

def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss

def check_inplace_ops(obj, path=""):
    if isinstance(obj, torch.Tensor):
        for name in dir(obj):
            if name.endswith("_"):
                print(f"Inplace operation detected at {path}.{name}")
    elif isinstance(obj, torch.nn.Module):
        pass  # 跳过 torch.nn.Module 对象的 _parameters 和 _modules 属性
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            check_inplace_ops(item, path=f"{path}[{i}]")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            check_inplace_ops(value, path=f"{path}.{key}")
    elif hasattr(obj, "__dict__"):
        for name in dir(obj):
            if not name.startswith("_"):
                check_inplace_ops(getattr(obj, name), path=f"{path}.{name}")

def find_threshold_micro(dev_yhat_raw, dev_y):
    dev_yhat_raw_1 = dev_yhat_raw.reshape(-1)
    dev_y_1 = dev_y.reshape(-1)
    sort_arg = np.argsort(dev_yhat_raw_1)
    sort_label = np.take_along_axis(dev_y_1, sort_arg, axis=0)
    label_count = np.sum(sort_label)
    correct = label_count - np.cumsum(sort_label)
    predict = dev_y_1.shape[0] + 1 - np.cumsum(np.ones_like(sort_label))
    f1 = 2 * correct / (predict + label_count)
    sort_yhat_raw = np.take_along_axis(dev_yhat_raw_1, sort_arg, axis=0)
    f1_argmax = np.argmax(f1)
    threshold = sort_yhat_raw[f1_argmax]
    # print(threshold)
    return threshold

class ClassifiyModule(nn.Module):
    def     __init__(self, Bertmodel_path, Bertmodel_config):
        super(ClassifiyModule, self).__init__()

        self.PreBert = BertModel.from_pretrained(Bertmodel_path, config=Bertmodel_config)
        # for parameter in self.PreBert.parameters():
        #     parameter.requires_grad = False

        self.model = args.model
        if args.model == 'bert':
            self.hidden_model = 1024
        if args.model == 'longformer':
            self.hidden_model = 768
        self.d_a = args.d_a
        self.n_labels = args.n_labels
        # 使用match
        self.w_linear = nn.Linear(self.hidden_model, self.hidden_model, bias=True)
        self.b_linear = nn.Linear(self.hidden_model, 1, bias=True)
        self.one = nn.Linear(args.for_num, 1)
        # self.u_reduce = nn.Linear(args.define_length, 1, bias=True)
        self.match_third_linears = nn.Linear(512, self.n_labels, bias=True)
        # self.label_fea = lableLSTM()
        # self.label_fea_tra = nn.Linear(1024, self.hidden_model)
        #
        # self.model_lstm = ModelLSTM()
        # self.label_fea_tra = nn.Linear(1024, self.hidden_model)

        self.clssificion = nn.Linear(self.hidden_model, self.n_labels, bias=True)

        self.dropout = nn.Dropout(p=0.2)

        self.first_linears = nn.Linear(self.hidden_model, self.d_a, bias=False)
        self.second_linears = nn.Linear(self.d_a, self.n_labels, bias=False)

        self.third_linears = nn.Linear(args.max_length, self.n_labels, bias=True)
        self.third_linears_teacher = nn.Linear(args.max_length, self.n_labels, bias=True)
        self.third_linears_student = nn.Linear(args.max_length, self.n_labels, bias=True)

        self.clssificion = nn.Linear(1024, self.n_labels, bias=True)

        #HAT
        self.HAT = HATM()

        # shared attention
        self.dropout_attn = 0.1
        self.share_attn_emb = nn.Embedding(self.n_labels, self.hidden_model)
        self.share_attn = nn.MultiheadAttention(embed_dim=self.hidden_model, num_heads=1,
                                                dropout=self.dropout_attn)
        self.label_attn = nn.MultiheadAttention(embed_dim=self.hidden_model, num_heads=1,
                                                dropout=self.dropout_attn)
        self.share_classify = nn.Linear(self.hidden_model, self.n_labels, bias=True)

        # Tran layer
        self.tran_layer = nn.Linear(self.n_labels,self.hidden_model)

        self.q_linear = nn.Linear(self.hidden_model,args.max_length)
        self.b = nn.Linear(self.hidden_model, 1)

        self.luck_relu = nn.LeakyReLU(negative_slope=0.1)

        self.cross_attention = CrossAttention(self.hidden_model)

        #laat
        self.third_linears_laat = nn.Linear(self.hidden_model, self.n_labels, bias=True)

        self.fusion = CrossAttention_fusion(self.hidden_model)

        self.mlp = MLP(input_dim=self.hidden_model, hidden_dim=self.hidden_model, output_dim=self.hidden_model,num_layers=2)
        self.b_mlp = MLP(input_dim=self.hidden_model, hidden_dim=self.hidden_model, output_dim=self.hidden_model,num_layers=2)

        self.label1 = nn.Linear(self.hidden_model,self.hidden_model)

    def encode(self, h_src, train, labels):
        # #去掉共享注意力
        # scroe,_ = self.share_attn(self.share_attn_emb.weight.unsqueeze(1).repeat(1, int(args.batch_size / len(args.gpus)), 1),h_src.transpose(0, 1),h_src.transpose(0, 1))
        # scroe = scroe.transpose(0, 1)

        # 使用共享注意力
        scroe = self.cross_attention(self.share_attn_emb.weight.unsqueeze(1).repeat(1, int(args.batch_size / len(args.gpus)), 1),h_src,h_src)

        # # 去掉共享注意力
        # scroe = self.cross_attention(labels.unsqueeze(0).repeat(h_src.size(0), 1, 1).transpose(0,1),h_src,h_src)

        HAT_score_student, HAT_score_teacher = self.HAT(scroe)
        fusion_score = self.fusion(HAT_score_student, HAT_score_teacher, HAT_score_teacher)

        b = self.b_mlp(labels)
        # use HAT
        if train == 'True':
            # h_teacher = HAT_score_teacher @ h_src.transpose(1,2)
            h_teacher = self.mlp(HAT_score_teacher)
            h_teacher_pred = contract('blh,lh->bl', h_teacher, b)
            # h_teacher_pred = self.third_linears.weight.mul(h_teacher).sum(dim=2).add(self.third_linears.bias)

            xx = self.mlp(fusion_score)
            h_student = self.mlp(xx)

            h_student_pred = contract('blh,lh->bl', h_student, b)
            # h_student_pred = self.third_linears.weight.mul(h_student).sum(dim=2).add(self.third_linears.bias)

        else:
            # h_teacher = HAT_score_teacher @ h_src.transpose(1, 2)
            xx = self.mlp(fusion_score)
            h_student = self.mlp(xx)
            h_student_pred = contract('blh,lh->bl', h_student, b)

            # h_student = xx @ h_src.transpose(1, 2)
            # h_student_pred = self.third_linears.weight.mul(h_student).sum(dim=2).add(self.third_linears.bias)
            h_teacher_pred = h_student_pred

        # # 去掉使用HAT
        # h_student = scroe @ h_src.transpose(1, 2)
        # h_student_pred = self.third_linears.weight.mul(h_student).sum(dim=2).add(self.third_linears.bias)
        # h_teacher_pred = h_student_pred
        return h_student_pred, h_teacher_pred, torch.mean(h_student[:,0:7,:],-1), h_student

    def _compute_cos(self, XS, XQ):
        '''
            Compute the pairwise cos distance
            @param XS (support x): support_size x ebd_dim
            @param XQ (support x): query_size x ebd_dim

            @return dist: query_size support_size

        '''
        # dot = torch.matmul(
        #     XS.unsqueeze(0).unsqueeze(-2),
        #     XQ.unsqueeze(1).unsqueeze(-1)
        # )

        dot = torch.matmul(XQ,XS.transpose(0,1))
        dot = dot.squeeze(-1).squeeze(-1)

        scale = (torch.norm(XS, dim=1) * torch.norm(XQ, dim=1))

        scale = torch.max(scale,torch.ones_like(scale) * 1e-8)

        dist = 1 - dot / scale

        return dist.sum()

    def forward(self, batch=None, token_type_ids=None, label=None, label_mask=None, return_dict=None,train='False'):
        input_ids = batch[0]
        attention_mask = batch[1]
        if len(batch) > 2:
            y = batch[2]
        else:
            y = None

        if self.model == 'bert':
            input_ids = input_ids.reshape(-1, args.max_length)
            attention_mask = attention_mask.reshape(-1, args.max_length)
        else:
            input_ids = input_ids.reshape(input_ids.size(0),-1)
            attention_mask = attention_mask.reshape(attention_mask.size(0), -1)

        x_student = self.PreBert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                 return_dict=return_dict)
        # h_src = self.dropout(x_student[0])
        h_src = x_student[0]


        # #使用laat
        # weight = F.tanh(self.first_linears(h_src))
        # attention_weight = self.second_linears(weight)
        # attention_weight = F.softmax(attention_weight,1).transpose(1,2)
        # weight_output = attention_weight @ h_src
        # h_student_pred = self.third_linears_laat.weight.mul(weight_output).sum(dim=2).add(self.third_linears_laat.bias)


        # #对标签表示进行操作
        # label = label.max(0)[0].max(1)[0]
        # label_att = label  # 7*768
        # if train == 'True':
        #     label[label_mask, :] = 0
        # # Tran Layer
        # Tran_label = self.tran_layer.weight @ label
        # Tran_label = label @ Tran_label
        # h_share_attn_stu, alpha_stu = self.share_attn(
        #     Tran_label.unsqueeze(1).repeat(1, int(args.batch_size / len(args.gpus)), 1), h_src.transpose(0, 1),
        #     h_src.transpose(0, 1))
        # h_share_attn_stu = h_share_attn_stu.transpose(0, 1)
        # discriminative_loss = 0.0
        # dis_label = label_att @ (self.tran_layer.weight @ label_att)
        # for j in range(args.n_labels):
        #     for k in range(args.n_labels):
        #         if j != k:
        #             # if Tran_label[j][0] == 0 or Tran_label[k][0] == 0:
        #             #     continue
        #             sim = -self._compute_cos(dis_label[j].reshape(-1, 1),
        #                                      dis_label[k].reshape(-1, 1))
        #             discriminative_loss = discriminative_loss + sim
        #
        # share_pred_stu = self.share_classify.weight.mul(h_share_attn_stu).sum(dim=2).add(self.share_classify.bias)

        # 对标签表示进行操作
        label = torch.max(label.view(args.n_labels,args.define_length,self.hidden_model),dim=1)[0]
            # label.max(0)[0].max(1)[0]
        label_linear = self.label1(label)

        tran_w = self.tran_layer.weight @ label_linear

        # #求label的伪逆
        # p_t = torch.mm(label.t(),label)
        # tran_w = p_t
        # #导致loss回传有问题，需要先复制
        # A = p_t
        # # 检查是否有无穷大的元素，并替换为0
        # A[torch.isinf(A)] = 0
        # # 对数据进行标准化处理
        # A_normalized = (A - A.mean()) / A.std()
        # U, S, V = torch.svd(A_normalized)
        # S_pseudo = torch.diag(torch.reciprocal(S))
        # A_pseudo = torch.mm(torch.mm(V.t(), S_pseudo), U.t())
        # A_pseudo_p = torch.mm(A_pseudo,label.t())
        # tran_w = torch.mm(A_pseudo_p,label_linear)

        y_mask = torch.zeros(y.size(0), y.size(1), dtype=torch.long).to(label.device)
        y_mask[:,label_mask] = 1
        label = self.share_attn_emb(y_mask).max(0)[0] + label_linear
        ham_label = label

        # Tran Layer
        Tran_label = torch.mm(label, tran_w)

        # # #去掉Tran layer部分
        # Tran_label = label_linear
        # ham_label = label_linear

        if train == 'True':
            label_clone = label.clone()
            label_clone[label_mask, :] = 0
            Tran_label = torch.mm(label_clone, tran_w)

        h_share_attn_stu = self.cross_attention(Tran_label.unsqueeze(1).repeat(1, int(args.batch_size / len(args.gpus)), 1),h_src, h_src)

        discriminative_loss = 0.0

        # dis_label = F.sigmoid(label_att @ (self.tran_layer.weight @ label_att))
        dis_label = torch.mm(ham_label, tran_w)
        # 计算 dis_label 之间的余弦相似度矩阵
        # 将 dis_label_tensor 进行 reshape 使得其变成二维矩阵
        dis_label_tensor = dis_label.reshape(args.n_labels, -1)
        # 使用 torch.unsqueeze 来添加一个维度，并利用广播机制进行计算
        cos_sim_matrix = self._compute_cos(dis_label_tensor, dis_label_tensor)
        discriminative_loss = cos_sim_matrix.sum().to(dis_label.device)

        share_pred_stu = self.share_classify.weight.mul(h_share_attn_stu).sum(dim=2).add(self.share_classify.bias)

        # #去掉tran layer
        # h_share_attn_stu, alpha_stu = self.share_attn(
        #     label.unsqueeze(1).repeat(1, int(args.batch_size / len(args.gpus)), 1), h_src.transpose(0, 1),
        #     h_src.transpose(0, 1))
        # h_share_attn_stu = h_share_attn_stu.transpose(0, 1)
        # share_pred_stu = self.share_classify.weight.mul(h_share_attn_stu).sum(dim=2).add(self.share_classify.bias)


        # 主体部分
        h_student_pred, h_teacher_pred, t_sne_hidden, attention_view = self.encode(h_src, train, ham_label)

        # #去掉Tran layer部分
        # h_share_attn_stu, alpha_stu = self.share_attn(
        #     label.unsqueeze(1).repeat(1, int(args.batch_size / len(args.gpus)), 1), h_src.transpose(0, 1),
        #     h_src.transpose(0, 1))
        # h_share_attn_stu = h_share_attn_stu.transpose(0, 1)
        # share_pred_stu = self.share_classify.weight.mul(h_share_attn_stu).sum(dim=2).add(self.share_classify.bias)

        # # 仅仅使用分类层
        # h_student_pred = h_classificy
        # share_pred_stu = h_classificy
        # h_teacher_pred = h_classificy
        # distances = h_student_pred

        # check_inplace_ops(locals())

        if train == 'True':
            if args.threshold == None:
                threshold = find_threshold_micro(h_student_pred.cpu().detach().numpy(), y.cpu().detach().numpy())
            else:
                threshold = args.threshold
        else:
            if args.threshold == None:
                threshold = find_threshold_micro(h_student_pred.cpu().detach().numpy(), y.cpu().detach().numpy())
            else:
                threshold = args.threshold

        # # 找到每行的最大值的索引
        # h_student_pred_cpu = h_student_pred.cpu().detach().numpy()
        # # 找到每行的最大值的索引，并转换为整数数组
        # max_indices = np.argmax(h_student_pred_cpu, axis=1)
        # max_indices = np.array(max_indices, dtype=np.int64)
        # # 创建一个与原始张量相同形状和数据类型的全零张量
        # result = np.zeros_like(h_student_pred_cpu, dtype=np.float64)
        # # 将每行最大值的索引处设为 1
        # result[np.arange(int(args.batch_size / len(args.gpus))), max_indices] = 1
        # result = torch.from_numpy(result).to(h_student_pred.to(h_student_pred.device))
        # h_student_pred = result
        # yhat = h_student_pred >= threshold

        yhat = h_student_pred >= threshold

        # return {"yhat_raw": h_student_pred, "yhat": yhat, "y": y, 'disl_label_pred':share_pred_stu, 'yhat_raw_teacher':h_teacher_pred,'Tran_label':h_teacher_pred}, None
        # return {"yhat_raw": h_student_pred, "yhat": yhat, "y": y,}, None
        return {"yhat_raw": h_student_pred, "yhat": yhat, "y": y, 'disl_label_pred': share_pred_stu, 'yhat_raw_teacher': h_teacher_pred, 'Tran_label': dis_label, 't_sne_hidden':t_sne_hidden, 'attention_view':attention_view}, discriminative_loss
