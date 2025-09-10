import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from topk.svm import SmoothTop1SVM
from models.mobilenetv4 import mobilenetv4_conv_tiny
import logging


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class Attn_Net(nn.Module):
    def __init__(self, L=512, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]
        if dropout:
            self.module.append(nn.Dropout(0.25))
        self.module.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x


class KAN_Attn_Net(nn.Module):
    def __init__(self, L=512, D=256, dropout=False, n_classes=1, degree=5, init_type='xavier'):
        super(KAN_Attn_Net, self).__init__()
        self.module = [
            ChebyKAN(width=[L, D], degree=degree, init_type=init_type),
            nn.Tanh()]
        if dropout:
            self.module.append(nn.Dropout(0.25))
        self.module.append(ChebyKAN(width=[D, n_classes], degree=degree, init_type=init_type))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=512, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x


class KAN_Attn_Net_Gated(nn.Module):
    def __init__(self, L=512, D=256, dropout=False, n_classes=1, degree=5, init_type='xavier'):
        super(KAN_Attn_Net_Gated, self).__init__()
        self.attention_a = [
            ChebyKAN(width=[L, D], degree=degree, init_type=init_type),
            nn.Tanh()]
        self.attention_b = [
            ChebyKAN(width=[L, D], degree=degree, init_type=init_type),
            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = ChebyKAN(width=[D, n_classes], degree=degree, init_type=init_type)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x


class CLAM_SB(nn.Module):
    def __init__(self, I=512, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [I, 512, 256], "big": [I, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False):
        assert len(h.shape) == 3
        B = h.shape[0]
        device = h.device
        A, h = self.attention_net(h)
        A = torch.permute(A, (0, 2, 1))
        A = F.softmax(A, dim=-1)

        if instance_eval:
            total_inst_loss = [0.0 for _ in range(B)]
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:
                    tmp_all_preds, tmp_all_targets = [], []
                    for b in range(B):
                        instance_loss, preds, targets = self.inst_eval(A[b, :, :], h[b], classifier)
                        total_inst_loss[b] += instance_loss
                else:
                    if self.subtyping:
                        for b in range(B):
                            instance_loss, preds, targets = self.inst_eval_out(A[b, :, :], h[b], classifier)
                            total_inst_loss[b] += instance_loss
                    else:
                        continue
            total_inst_loss = torch.stack(total_inst_loss, 0)
            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.bmm(A, h)
        logits = self.classifiers(M)

        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})

        return logits.squeeze(1), A, results_dict, M


class CLAM_MB(CLAM_SB):
    def __init__(self, I=512, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, dim_reduce=True):
        nn.Module.__init__(self)
        self.size_dict = {"small": [I, 512, 256], "big": [I, 512, 384]}
        size = self.size_dict[size_arg]
        if dim_reduce:
            fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        else:
            fc = []
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        initialize_weights(self)

    def forward(self, h, label=None, instance_eval=False, return_features=False):
        assert len(h.shape) == 3
        B = h.shape[0]
        device = h.device
        A, h = self.attention_net(h)
        A = torch.permute(A, (0, 2, 1))
        A = F.softmax(A, dim=-1)

        if instance_eval:
            total_inst_loss = [0.0 for _ in range(B)]
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:
                    for b in range(B):
                        instance_loss, preds, targets = self.inst_eval(A[b, i, :], h[b], classifier)
                        total_inst_loss[b] += instance_loss
                else:
                    if self.subtyping:
                        for b in range(B):
                            instance_loss, preds, targets = self.inst_eval_out(A[b, i, :], h[b], classifier)
                            total_inst_loss[b] += instance_loss
                    else:
                        continue
            total_inst_loss = torch.stack(total_inst_loss, 0)
            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.bmm(A, h)
        logits = torch.empty(B, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[:, c] = self.classifiers[c](M[:, c, :]).squeeze(1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, A, results_dict, M


class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, init_type='xavier'):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        if init_type == 'normal':
            nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        elif init_type == 'xavier':
            nn.init.xavier_normal_(self.cheby_coeffs)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(self.cheby_coeffs)
        else:
            raise NotImplementedError
        logging.debug(
            f"ChebyKANLayer init: mean={self.cheby_coeffs.mean().item():.4f}, std={self.cheby_coeffs.std().item():.4f}")

    def forward(self, x):
        x = torch.tanh(x)
        if len(x.shape) == 2:
            cheby = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
            if self.degree > 0:
                cheby[:, :, 1] = x
            for i in range(2, self.degree + 1):
                cheby[:, :, i] = 2 * x * cheby[:, :, i - 1].clone() - cheby[:, :, i - 2].clone()
            y = torch.einsum('bid,iod->bo', cheby, self.cheby_coeffs)
            return y
        elif len(x.shape) == 3:
            cheby = torch.ones(x.shape[0], x.shape[1], self.inputdim, self.degree + 1, device=x.device)
            if self.degree > 0:
                cheby[:, :, :, 1] = x
            for i in range(2, self.degree + 1):
                cheby[:, :, :, i] = 2 * x * cheby[:, :, :, i - 1].clone() - cheby[:, :, :, i - 2].clone()
            y = torch.einsum('bnid,iod->bno', cheby, self.cheby_coeffs)
            return y


class ChebyKAN(nn.Module):
    def __init__(self, width=None, degree=5, init_type='xavier'):
        super(ChebyKAN, self).__init__()
        self.width = width
        self.depth = len(width) - 1
        act_fun = []
        for l in range(self.depth):
            act_fun.append(ChebyKANLayer(self.width[l], self.width[l + 1], degree, init_type=init_type))
        self.act_fun = nn.ModuleList(act_fun)
        self.degree = degree

    def forward(self, x):
        for l in range(self.depth):
            x = self.act_fun[l](x)
        return x


def _gumbel_sigmoid(logits, tau=0.5, hard=False, eps=1e-10, training=True, threshold=0.3):
    if training:
        gumbels1 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels2 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
    else:
        y_soft = logits.sigmoid()
    if hard:
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).masked_fill(y_soft > threshold, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


class KAN_CLAM_MB_v4(CLAM_MB):
    def __init__(self, I=512, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
                 n_classes_1=2, instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, dim_reduce=True, args=None):
        nn.Module.__init__(self)
        self.size_dict = {"small": [I, 512, 256], "big": [I, 512, 384]}
        size = self.size_dict[size_arg]
        if dim_reduce:
            fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
            fc_1 = [nn.Linear(size[0], size[1]), nn.ReLU()]  # 为label_1创建独立的特征变换
        else:
            fc = []
            fc_1 = []
        if dropout:
            fc.append(nn.Dropout(0.25))
            fc_1.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
            attention_net_1 = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes_1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
            attention_net_1 = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes_1)
        fc.append(attention_net)
        fc_1.append(attention_net_1)
        self.attention_net = nn.Sequential(*fc)
        self.attention_net_1 = nn.Sequential(*fc_1)

        bag_classifiers = [
            ChebyKAN(width=[size[1], 1], degree=args.degree, init_type=args.init_type) for i in range(n_classes)
        ]
        bag_classifiers_1 = [
            ChebyKAN(width=[size[1], 1], degree=args.degree, init_type=args.init_type) for i in range(n_classes_1)
        ]
        self.classifiers = nn.ModuleList(bag_classifiers)
        self.classifiers_1 = nn.ModuleList(bag_classifiers_1)
        instance_classifiers = [
            ChebyKAN(width=[size[1], 2], degree=args.degree, init_type=args.init_type) for i in range(n_classes)
        ]
        instance_classifiers_1 = [
            ChebyKAN(width=[size[1], 2], degree=args.degree, init_type=args.init_type) for i in range(n_classes_1)
        ]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.instance_classifiers_1 = nn.ModuleList(instance_classifiers_1)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.n_classes_1 = n_classes_1
        self.subtyping = subtyping
        initialize_weights(self)

    def forward(self, h, label=None, label_1=None, instance_eval=False, return_features=False):
        assert len(h.shape) == 3
        B, N = h.shape[0], h.shape[1]
        device = h.device

        h_1 = h.clone()  # 为label_1复制一份独立的输入

        attns, h = self.attention_net(h)
        A = torch.permute(attns, (0, 2, 1))
        A = F.softmax(A, dim=-1)

        attns_1, h_1 = self.attention_net_1(h_1)  # 使用独立的attention_net_1处理h_1
        A_1 = torch.permute(attns_1, (0, 2, 1))
        A_1 = F.softmax(A_1, dim=-1)

        if instance_eval:
            total_inst_loss = [0.0 for _ in range(B)]
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:
                    for b in range(B):
                        instance_loss, preds, targets = self.inst_eval(A[b, i, :], h[b], classifier)
                        total_inst_loss[b] += instance_loss
                else:
                    if self.subtyping:
                        for b in range(B):
                            instance_loss, preds, targets = self.inst_eval_out(A[b, i, :], h[b], classifier)
                            total_inst_loss[b] += instance_loss
                    else:
                        continue
            total_inst_loss = torch.stack(total_inst_loss, 0)
            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

            total_inst_loss_1 = [0.0 for _ in range(B)]
            inst_labels_1 = F.one_hot(label_1, num_classes=self.n_classes_1).squeeze()
            for i in range(len(self.instance_classifiers_1)):
                inst_label_1 = inst_labels_1[i].item()
                classifier_1 = self.instance_classifiers_1[i]
                if inst_label_1 == 1:
                    for b in range(B):
                        instance_loss_1, preds_1, targets_1 = self.inst_eval(A_1[b, i, :], h_1[b], classifier_1)
                        total_inst_loss_1[b] += instance_loss_1
                else:
                    if self.subtyping:
                        for b in range(B):
                            instance_loss_1, preds_1, targets_1 = self.inst_eval_out(A_1[b, i, :], h_1[b], classifier_1)
                            total_inst_loss_1[b] += instance_loss_1
                    else:
                        continue
            total_inst_loss_1 = torch.stack(total_inst_loss_1, 0)
            if self.subtyping:
                total_inst_loss_1 /= len(self.instance_classifiers_1)

        if self.training:
            mask = _gumbel_sigmoid(attns, tau=0.5, hard=True, eps=1e-10, training=True, threshold=0.3)
            mask_add = mask[:, :, 0] + mask[:, :, 1]
            mask_union = torch.zeros_like(mask_add).masked_fill(mask_add != 0, 1.0)
            mask_rate = (mask_union - mask_add.detach() + mask_add).sum() / (B * N)
            select_exp_attns = torch.exp(attns) * mask
            select_A = (select_exp_attns / torch.sum(select_exp_attns)).permute(0, 2, 1)
            select_M = torch.bmm(select_A, h)
            select_logits = torch.empty(B, self.n_classes).float().to(device)
            for c in range(self.n_classes):
                select_logits[:, c] = self.classifiers[c](select_M[:, c, :]).squeeze(1)
            M = torch.bmm(A, h)
            logits = torch.empty(B, self.n_classes).float().to(device)
            for c in range(self.n_classes):
                logits[:, c] = self.classifiers[c](M[:, c, :]).squeeze(1)

            mask_1 = _gumbel_sigmoid(attns_1, tau=0.5, hard=True, eps=1e-10, training=True, threshold=0.3)
            mask_add_1 = mask_1[:, :, 0] + mask_1[:, :, 1] if self.n_classes_1 > 1 else mask_1[:, :, 0]
            mask_union_1 = torch.zeros_like(mask_add_1).masked_fill(mask_add_1 != 0, 1.0)
            mask_rate_1 = (mask_union_1 - mask_add_1.detach() + mask_add_1).sum() / (B * N)
            select_exp_attns_1 = torch.exp(attns_1) * mask_1
            select_A_1 = (select_exp_attns_1 / torch.sum(select_exp_attns_1)).permute(0, 2, 1)
            select_M_1 = torch.bmm(select_A_1, h_1)
            select_logits_1 = torch.empty(B, self.n_classes_1).float().to(device)
            M_1 = torch.bmm(A_1, h_1)
            logits_1 = torch.empty(B, self.n_classes_1).float().to(device)
            for c in range(self.n_classes_1):
                select_logits_1[:, c] = self.classifiers_1[c](select_M_1[:, c, :]).squeeze(1)
                logits_1[:, c] = self.classifiers_1[c](M_1[:, c, :]).squeeze(1)

            results_dict = {'instance_loss': total_inst_loss} if instance_eval else {}
            results_dict_1 = {'instance_loss_1': total_inst_loss_1} if instance_eval else {}
            if return_features:
                results_dict.update({'features': M})
                results_dict_1.update({'features_1': M_1})

            logging.info(f"Attns mean={attns.mean().item():.4f}, std={attns.std().item():.4f}, "
                         f"Mask mean={mask.mean().item():.4f}, Attns_1 mean={attns_1.mean().item():.4f}, Mask_1 mean={mask_1.mean().item():.4f}")
            return logits, A, results_dict, M, select_logits, select_M, mask_rate, logits_1, A_1, results_dict_1, M_1, select_logits_1, select_M_1, mask_rate_1

        else:
            mask = _gumbel_sigmoid(attns, tau=0.3, hard=True, eps=1e-10, training=False, threshold=0.2)
            select_exp_attns = torch.exp(attns) * mask
            select_logits = torch.empty(B, self.n_classes).float().to(device)
            for c in range(self.n_classes):
                select_tokens = h[:, mask[0, :, c] == 1, :]
                select_exp_attns_ = select_exp_attns[:, mask[0, :, c] == 1, c]
                select_A_ = (select_exp_attns_ / (torch.sum(select_exp_attns_) + 1e-16))
                select_M_ = torch.mm(select_A_, select_tokens[0])
                select_logits[:, c] = self.classifiers[c](select_M_[:, :]).squeeze(1)

            mask_1 = _gumbel_sigmoid(attns_1, tau=0.3, hard=True, eps=1e-10, training=False, threshold=0.2)
            select_exp_attns_1 = torch.exp(attns_1) * mask_1
            select_logits_1 = torch.empty(B, self.n_classes_1).float().to(device)
            for c in range(self.n_classes_1):
                select_tokens_1 = h_1[:, mask_1[0, :, c] == 1, :]
                select_exp_attns__1 = select_exp_attns_1[:, mask_1[0, :, c] == 1, c]
                select_A__1 = (select_exp_attns__1 / (torch.sum(select_exp_attns__1) + 1e-16))
                select_M__1 = torch.mm(select_A__1, select_tokens_1[0])
                select_logits_1[:, c] = self.classifiers_1[c](select_M__1[:, :]).squeeze(1)

            logging.info(f"Test Attns mean={attns.mean().item():.4f}, std={attns.std().item():.4f}, "
                         f"Mask mean={mask.mean().item():.4f}, Attns_1 mean={attns_1.mean().item():.4f}, Mask_1 mean={mask_1.mean().item():.4f}")
            return select_logits, select_logits_1


class KAN_CLAM_MB_v5(CLAM_MB):
    def __init__(self, I=512, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
                 n_classes_1=2, instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, dim_reduce=True, args=None):
        nn.Module.__init__(self)
        self.size_dict = {"small": [I, 512, 256], "big": [I, 512, 384]}
        size = self.size_dict[size_arg]
        if dim_reduce:
            fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        else:
            fc = []
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [
            ChebyKAN(width=[size[1], 1], degree=args.degree, init_type=args.init_type) for i in range(n_classes)
        ]
        bag_classifiers_1 = [
            ChebyKAN(width=[size[1], 1], degree=args.degree, init_type=args.init_type) for i in range(n_classes_1)
        ]
        self.classifiers = nn.ModuleList(bag_classifiers)
        self.classifiers_1 = nn.ModuleList(bag_classifiers_1)
        instance_classifiers = [
            ChebyKAN(width=[size[1], 2], degree=args.degree, init_type=args.init_type) for i in range(n_classes)
        ]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.n_classes_1 = n_classes_1
        self.subtyping = subtyping
        initialize_weights(self)

    def forward(self, h, scores=None, only_attn=False):
        assert len(h.shape) == 3
        B = h.shape[0]
        device = h.device
        if scores is None:
            attns, h = self.attention_net(h)
            if only_attn:
                return attns
            mask = _gumbel_sigmoid(attns, tau=0.5, hard=True, eps=1e-10, training=False, threshold=0.3)
            select_exp_attns = torch.exp(attns) * mask
        else:
            mask = _gumbel_sigmoid(scores, tau=0.5, hard=True, eps=1e-10, training=False, threshold=0.3)
        select_logits = torch.empty(B, self.n_classes).float().to(device)
        select_logits_1 = torch.empty(B, self.n_classes_1).float().to(device)
        for c in range(self.n_classes):
            if scores is None:
                select_tokens = h[:, mask[0, :, c] == 1, :]
                select_exp_attns_ = select_exp_attns[:, mask[0, :, c] == 1, c]
                select_A_ = (select_exp_attns_ / (torch.sum(select_exp_attns_) + 1e-16))
            else:
                select_patches = h[:, mask[0, :, c] == 1, :]
                attns, select_tokens = self.attention_net(select_patches)
                select_exp_attns_ = torch.exp(attns[:, :, c])
                select_A_ = (select_exp_attns_ / (torch.sum(select_exp_attns_) + 1e-16))
            select_M_ = torch.mm(select_A_, select_tokens[0])
            select_logits[:, c] = self.classifiers[c](select_M_[:, :]).squeeze(1)
        for c in range(self.n_classes_1):
            if scores is None:
                select_tokens = h[:, mask[0, :, c % self.n_classes] == 1, :]
                select_exp_attns_ = select_exp_attns[:, mask[0, :, c % self.n_classes] == 1, c % self.n_classes]
                select_A_ = (select_exp_attns_ / (torch.sum(select_exp_attns_) + 1e-16))
            else:
                select_patches = h[:, mask[0, :, c % self.n_classes] == 1, :]
                attns, select_tokens = self.attention_net(select_patches)
                select_exp_attns_ = torch.exp(attns[:, :, c % self.n_classes])
                select_A_ = (select_exp_attns_ / (torch.sum(select_exp_attns_) + 1e-16))
            select_M_ = torch.mm(select_A_, select_tokens[0])
            select_logits_1[:, c] = self.classifiers_1[c](select_M_[:, :]).squeeze(1)
        return select_logits, select_logits_1, mask


class ImageProcessor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.lwc == 'mbv4t':
            self.extractor = mobilenetv4_conv_tiny(n_classes=args.n_classes)
        else:
            raise NotImplementedError

    def forward(self, img, discrete=True):
        img = img.permute(0, 1, 4, 2, 3).float() / 255. - 0.5
        B, N, C, H, W = img.shape
        assert B == 1
        img = img.reshape(B * N, C, H, W)
        scores = self.extractor(img).reshape(B, N, self.args.n_classes)
        if discrete:
            scores = _gumbel_sigmoid(scores, tau=0.5, hard=True, eps=1e-10, training=False, threshold=0.3)
        return scores