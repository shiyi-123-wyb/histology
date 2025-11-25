# ===== models/hdmil_hier.py =====
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()


class Attn_Net_Gated(nn.Module):
    """
    标准 gated attention 模块：Linear → Tanh / Sigmoid → 逐元素乘 → Linear
    """
    def __init__(self, L=512, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()
        ]
        self.attention_b = [
            nn.Linear(L, D),
            nn.Sigmoid()
        ]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        """
        x: [N, L] 或 [K, L]
        返回:
            A: [N, n_classes]
            x: 原特征（为了和旧接口一致一起返回）
        """
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b)
        return A, x


class HierarchicalMILModel(nn.Module):
    """
    Cluster-wise Hierarchical MIL

    输入:
        h: [B, N, I]         patch-level 特征
        cluster_ids: [B, N]  每个 patch 的 cluster index

    结构:
        1) cluster embedding 通过 concat + Linear 融合回 patch 特征
        2) Intra-cluster attention:  patch -> cluster feature
        3) Inter-cluster attention: cluster feature -> slide feature
        4) Classifier(+Dropout): slide feature -> logits
    """

    def __init__(self,
                 I=512,
                 num_clusters=10,
                 n_classes=2,
                 dropout=True,
                 cluster_emb_dim=8,
                 use_cluster_emb=True,
                 max_patches_per_cluster=None,
                 slide_dropout=0.0):
        super().__init__()
        self.num_clusters = num_clusters
        self.n_classes = n_classes
        self.input_dim = I

        # ====== 1. cluster embedding ======
        self.cluster_emb_dim = cluster_emb_dim
        self.use_cluster_emb = use_cluster_emb
        self.cluster_embedding = nn.Embedding(
            num_embeddings=num_clusters,
            embedding_dim=cluster_emb_dim
        )
        nn.init.normal_(self.cluster_embedding.weight, mean=0.0, std=0.02)

        # 利用 concat + Linear 融合 [h, emb] -> I 维
        self.cluster_fusion = nn.Sequential(
            nn.Linear(I + cluster_emb_dim, I),
            nn.ReLU(inplace=True)
        )

        # 每个簇最多使用的 patch 数；None 表示不限制
        self.max_patches_per_cluster = max_patches_per_cluster

        # ====== 2. cluster 内 attention：patch 级 → cluster 级 ======
        self.intra_attn = Attn_Net_Gated(L=I, D=256, dropout=dropout, n_classes=1)
        self.intra_fc = nn.Sequential(
            nn.Linear(I, 512),
            nn.ReLU(inplace=True)
        )

        # ====== 3. cluster 间 attention：cluster 级 → slide 级 ======
        self.inter_attn = Attn_Net_Gated(L=512, D=256, dropout=dropout, n_classes=1)
        self.inter_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        # ====== 4. slide-level classifier (+Dropout) ======
        if slide_dropout and slide_dropout > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(slide_dropout),
                nn.Linear(256, n_classes)
            )
        else:
            self.classifier = nn.Linear(256, n_classes)

        initialize_weights(self)

    def _fuse_cluster_embedding(self, h, cluster_ids):
        """
        h: [B, N, I]
        cluster_ids: [B, N]
        return: h': [B, N, I]  融合 cluster embedding 后的特征
        """
        if (not self.use_cluster_emb) or (cluster_ids is None):
            return h

        B, N, I = h.shape
        device = h.device

        # [B, N, emb_dim]
        emb = self.cluster_embedding(cluster_ids)  # 0~num_clusters-1

        # concat: [B, N, I + emb_dim]
        x = torch.cat([h, emb], dim=-1)
        x = x.view(B * N, I + self.cluster_emb_dim)
        x = self.cluster_fusion(x)          # [B*N, I]
        x = x.view(B, N, I)
        return x

    def forward(self, h, cluster_ids):
        """
        h: [B, N, I]
        cluster_ids: [B, N]
        """
        assert len(h.shape) == 3
        B, N, I = h.shape
        device = h.device

        # ====== (1) 先融合 cluster embedding ======
        h = self._fuse_cluster_embedding(h, cluster_ids)

        slide_feats = []

        for b in range(B):
            h_b = h[b]                   # [N, I]
            clusters_b = cluster_ids[b]  # [N]
            cluster_feats = []

            # 只遍历实际出现的簇，避免空簇
            unique_cids = torch.unique(clusters_b)

            for cid in unique_cids:
                mask = (clusters_b == cid)
                num_patches = mask.sum().item()
                if num_patches == 0:
                    continue

                # ====== (2) 每个簇内如果 patch 太多，随机下采样（可选） ======
                if (self.max_patches_per_cluster is not None) and (num_patches > self.max_patches_per_cluster):
                    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
                    perm = torch.randperm(num_patches, device=device)[: self.max_patches_per_cluster]
                    idx = idx[perm]
                    sub_mask = torch.zeros_like(mask, dtype=torch.bool)
                    sub_mask[idx] = True
                    mask = sub_mask

                h_c = h_b[mask]          # [Nc, I]

                # --- intra-cluster attention ---
                A_c, h_c = self.intra_attn(h_c)   # A_c: [Nc, 1]
                A_c = torch.softmax(A_c, dim=0)   # weights over patches
                c_feat = torch.mm(A_c.T, h_c)     # [1, I]
                c_feat = self.intra_fc(c_feat)    # [1, 512]
                cluster_feats.append(c_feat)

            if len(cluster_feats) == 0:
                # 极端情况：该 slide 没有有效 patch
                slide_feats.append(torch.zeros(1, 256).to(device))
                continue

            cluster_feats = torch.cat(cluster_feats, dim=0)  # [K', 512]

            # --- inter-cluster attention ---
            A_slide, h_slide = self.inter_attn(cluster_feats)   # [K', 1], [K', 512]
            A_slide = torch.softmax(A_slide, dim=0)             # over clusters
            slide_feat = torch.mm(A_slide.T, h_slide)           # [1, 512]
            slide_feat = self.inter_fc(slide_feat)              # [1, 256]

            slide_feats.append(slide_feat)

        slide_feats = torch.cat(slide_feats, dim=0)   # [B, 256]
        logits = self.classifier(slide_feats)         # [B, n_classes]
        return logits
