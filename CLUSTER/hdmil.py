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
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class Attn_Net_Gated(nn.Module):
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
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b)
        return A, x


class HierarchicalMILModel(nn.Module):

    def __init__(self, I=512, num_clusters=10, n_classes=2, dropout=True, cluster_emb_dim=8, use_cluster_emb=True, max_patches_per_cluster=None,slide_dropout=0.0):
        super().__init__()
        self.num_clusters = num_clusters
        self.n_classes = n_classes
        self.input_dim = I

        self.cluster_emb_dim = cluster_emb_dim
        self.use_cluster_emb = use_cluster_emb
        self.cluster_embedding = nn.Embedding(
            num_embeddings=num_clusters,
            embedding_dim=cluster_emb_dim
        )
        nn.init.normal_(self.cluster_embedding.weight, mean=0.0, std=0.02)

        self.cluster_fusion = nn.Sequential(
            nn.Linear(I + cluster_emb_dim, I),
            nn.ReLU(inplace=True)
        )

        self.max_patches_per_cluster = max_patches_per_cluster

        # ====== 1. cluster 内 attention：patch 级 → cluster 级 ======
        self.intra_attn = Attn_Net_Gated(L=I, D=256, dropout=dropout, n_classes=1)
        self.intra_fc = nn.Sequential(
            nn.Linear(I, 512),
            nn.ReLU(inplace=True)
        )

        # ====== 2. cluster 间 attention：cluster 级 → slide 级 ======
        self.inter_attn = Attn_Net_Gated(L=512, D=256, dropout=dropout, n_classes=1)
        self.inter_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        # ====== 3. slide-level classifier (+Dropout) ======
        if slide_dropout and slide_dropout > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(slide_dropout),
                nn.Linear(256, n_classes)
            )
        else:
            self.classifier = nn.Linear(256, n_classes)

        initialize_weights(self)

    def _fuse_cluster_embedding(self, h, cluster_ids):
        if (not self.use_cluster_emb) or (cluster_ids is None):
            return h
        B, N, I = h.shape
        device = h.device
        emb = self.cluster_embedding(cluster_ids)  
        x = torch.cat([h, emb], dim=-1)
        x = x.view(B * N, I + self.cluster_emb_dim)
        x = self.cluster_fusion(x)         
        x = x.view(B, N, I)
        return x

    def forward(self, h, cluster_ids):
        assert len(h.shape) == 3
        B, N, I = h.shape
        device = h.device
        h = self._fuse_cluster_embedding(h, cluster_ids)

        slide_feats = []

        for b in range(B):
            h_b = h[b]                   
            clusters_b = cluster_ids[b]  
            cluster_feats = []
            unique_cids = torch.unique(clusters_b)

            for cid in unique_cids:
                mask = (clusters_b == cid)
                num_patches = mask.sum().item()
                if num_patches == 0:
                    continue

                if (self.max_patches_per_cluster is not None) and (num_patches > self.max_patches_per_cluster):
                    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
                    perm = torch.randperm(num_patches, device=device)[: self.max_patches_per_cluster]
                    idx = idx[perm]
                    sub_mask = torch.zeros_like(mask, dtype=torch.bool)
                    sub_mask[idx] = True
                    mask = sub_mask

                h_c = h_b[mask]         

                A_c, h_c = self.intra_attn(h_c)  
                A_c = torch.softmax(A_c, dim=0)   
                c_feat = torch.mm(A_c.T, h_c)   
                c_feat = self.intra_fc(c_feat)    
                cluster_feats.append(c_feat)

            if len(cluster_feats) == 0:
                slide_feats.append(torch.zeros(1, 256).to(device))
                continue

            cluster_feats = torch.cat(cluster_feats, dim=0) 

            A_slide, h_slide = self.inter_attn(cluster_feats)   
            A_slide = torch.softmax(A_slide, dim=0)            
            slide_feat = torch.mm(A_slide.T, h_slide)          
            slide_feat = self.inter_fc(slide_feat)             

            slide_feats.append(slide_feat)

        slide_feats = torch.cat(slide_feats, dim=0)  
        logits = self.classifier(slide_feats)        
        return logits
