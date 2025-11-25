# ===== DMIN.py (hierarchical, cluster-aware) =====
import os
import sys
import logging
from collections import Counter

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, SequentialSampler, Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pickle

sys.path.append('..')


class WSIDataset(Dataset):

    def __init__(self, args, wsi_labels, infold_cases, phase=None, target_cluster=None):
        self.args = args
        self.phase = phase
        self.target_cluster = target_cluster

        self.infold_features = []
        self.infold_labels = []
        self.infold_cluster_ids = []

        # ========= 加载 coord -> global_cluster 映射 =========
        if not hasattr(self.args, "coord_pkl_path"):
            raise ValueError("❌ WSIDataset 需要 args.coord_pkl_path，但当前 args 没有这个属性。")

        coord_map_path = self.args.coord_pkl_path
        print(f"[DMIN DEBUG] Trying to open coord_to_global_cluster.pkl from: {coord_map_path}")
        logging.info(f"[WSIDataset-{phase}] using coord_to_global_cluster from: {coord_map_path}")

        if not os.path.exists(coord_map_path):
            print(f"[DMIN ERROR] coord_to_global_cluster.pkl 不存在，请检查路径！\n>>> {coord_map_path}")
            raise FileNotFoundError(f"coord_to_global_cluster.pkl 文件不存在: {coord_map_path}")

        with open(coord_map_path, 'rb') as f:
            self.coord_to_cluster = pickle.load(f)
        print(f"[DMIN DEBUG] 成功加载 coord_to_global_cluster.pkl，共 {len(self.coord_to_cluster)} 条记录")

        # ========= 正常加载 patch features =========
        for case_id, slide_id, label in wsi_labels:
            if case_id not in infold_cases:
                continue

            h5_path = os.path.join(args.feature_dir, f"{slide_id}.h5")
            if not os.path.exists(h5_path):
                continue

            with h5py.File(h5_path, 'r') as f:
                feats = f['features'][:]  # (N, D)
                coords = f['coords'][:]   # (N, 2)

            cluster_ids = np.zeros((coords.shape[0]), dtype=np.int64)

            for i, coord in enumerate(coords):
                key = (int(coord[0]), int(coord[1]))
                if key in self.coord_to_cluster:
                    cluster_ids[i] = int(self.coord_to_cluster[key])
                else:
                    cluster_ids[i] = 0  # 没找到就丢到 0 类

            # 如果指定了 target_cluster，则只保留该簇的 patch
            if self.target_cluster is not None:
                mask = (cluster_ids == self.target_cluster)
                if mask.sum() == 0:
                    continue
                feats = feats[mask]
                cluster_ids = cluster_ids[mask]

                if feats.shape[0] < 10:
                    print(f"Skip {slide_id} in cluster {self.target_cluster}: only {feats.shape[0]} patches")
                    continue

            feats_tensor = torch.from_numpy(feats.astype(np.float32))
            cluster_ids_tensor = torch.from_numpy(cluster_ids.astype(np.int64))

            # 训练阶段打乱 bag 内 patch 顺序
            if self.phase == 'train':
                perm = torch.randperm(feats_tensor.shape[0])
                feats_tensor = feats_tensor[perm]
                cluster_ids_tensor = cluster_ids_tensor[perm]

            self.infold_features.append(feats_tensor)
            self.infold_labels.append(label)
            self.infold_cluster_ids.append(cluster_ids_tensor)

        print(f"[DMIN DEBUG] Loaded {len(self.infold_features)} WSIs for {phase}, "
              f"target_cluster={self.target_cluster}")

    def __len__(self):
        return len(self.infold_features)

    def __getitem__(self, idx):
        feats = self.infold_features[idx]
        label = torch.tensor(self.infold_labels[idx], dtype=torch.long)
        cluster_ids = self.infold_cluster_ids[idx]
        return feats, label, cluster_ids


class DMINMIL:
    def __init__(self, args):
        self.args = args

        if args.dataset in ['Camelyon16', 'TCGA-NSCLC', 'TCGA-BRCA', 'TCGA-RCC', 'CustomDataset']:
            self.train_loader, self.test_loader = self.init_data_wsi()
        else:
            raise NotImplementedError

        self.model = self.init_model()
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params / 1e6:.3f} M")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd)

        self.loss = torch.nn.CrossEntropyLoss(
            reduction='mean',
            label_smoothing=getattr(self.args, "label_smoothing", 0.0)
        )

        self.best_auc = -1.0
        self.best_test_metrics = None
        self.ckpt_name = os.path.join(self.args.ckpt_dir, 'best_epoch.pth')

        self.step = 0
        self.warmup_steps = 100

    def read_wsi_label(self):
        data = pd.read_csv(self.args.label_csv)
        logging.info(f"Label CSV head:\n{data.head().to_string()}")
        logging.info(f"Column names: {list(data.columns)}")

        wsi_labels = []
        for i in range(len(data)):
            if self.args.dataset == 'CustomDataset':
                case_id = str(data.loc[i, "ID"])
                slide_id = str(data.loc[i, "ID"])
                label = int(data.loc[i, "label"])
                if label not in [0, 1]:
                    raise ValueError(f"Invalid label {label} at row {i}, expect 0/1")
            else:
                case_id = data.loc[i, "case_id"]
                slide_id = data.loc[i, "slide_id"]
                label = data.loc[i, "label"]

            wsi_labels.append([case_id, slide_id, label])

        return wsi_labels

    def read_in_fold_cases(self, fold_csv):
        data = pd.read_csv(fold_csv)
        train_cases, valid_cases, test_cases = [], [], []

        for i in range(len(data)):
            train_val = data.loc[i, 'train']
            if pd.notna(train_val):
                train_cases.append(str(int(train_val)))

            val_val = data.loc[i, 'val']
            if pd.notna(val_val):
                valid_cases.append(str(int(val_val)))

            test_val = data.loc[i, 'test']
            if pd.notna(test_val):
                test_cases.append(str(int(test_val)))

        logging.info(f"Train cases: {len(train_cases)}, sample: {train_cases[:5]}")
        logging.info(f"Valid cases: {len(valid_cases)}, sample: {valid_cases[:5]}")
        logging.info(f"Test  cases: {len(test_cases)}, sample: {test_cases[:5]}")

        return train_cases, valid_cases, test_cases

    def make_weights_for_balanced_classes_split(self, data_set):
        N = float(len(data_set))
        classes = {}
        for label in data_set.infold_labels:
            if label not in classes:
                classes[label] = 1
            else:
                classes[label] += 1

        weight = [0.0] * int(N)
        for idx in range(len(data_set)):
            y = data_set.infold_labels[idx]
            weight[idx] = N / classes[y]

        return torch.DoubleTensor(weight)

    def init_data_wsi(self):

        wsi_labels = self.read_wsi_label()
        split_csv = os.path.join(self.args.split_dir, f'splits_{self.args.k}.csv')

        train_cases, valid_cases, test_cases = self.read_in_fold_cases(split_csv)
        train_val_cases = list(set(train_cases + valid_cases))

        self.train_cases = train_val_cases
        self.test_cases = test_cases
        self.wsi_labels = wsi_labels

        train_set = WSIDataset(self.args, wsi_labels, train_val_cases, phase='train', target_cluster=None)
        test_set = WSIDataset(self.args, wsi_labels, test_cases, phase='test', target_cluster=None)

        logging.info(
            f"Case/WSI number for trainset (train+valid) in fold-{self.args.k} = "
            f"{len(train_val_cases)}/{len(train_set)}"
        )
        logging.info(
            f"Case/WSI number for testset in fold-{self.args.k} = "
            f"{len(test_cases)}/{len(test_set)}"
        )

        if len(train_set) == 0:
            raise ValueError("Train set is empty. Check case_id matching or feature paths.")

        weights = self.make_weights_for_balanced_classes_split(train_set)

        train_loader = DataLoader(
            train_set,
            batch_size=1,
            sampler=WeightedRandomSampler(weights, len(weights), replacement=True)
        )
        test_loader = DataLoader(
            test_set,
            batch_size=1,
            sampler=SequentialSampler(test_set)
        )
        return train_loader, test_loader

    # ---------- 模型 ----------
    def init_model(self):
        # 注意：这里导的是你下面这个文件里的 HierarchicalMILModel
        from models.hdmil import HierarchicalMILModel

        mppc = getattr(self.args, "max_patches_per_cluster", None)
        if mppc is not None and mppc <= 0:
            mppc = None

        model = HierarchicalMILModel(
            I=self.args.feature_dim,
            num_clusters=self.args.num_clusters,
            n_classes=self.args.n_classes,
            dropout=True,
            cluster_emb_dim=8,                   # 比较安全的 embedding 维度
            use_cluster_emb=True,
            max_patches_per_cluster=mppc,        # 每簇最多 patch 数
            slide_dropout=getattr(self.args, "slide_dropout", 0.0)
        ).to(self.args.device)
        return model

    # ---------- 训练 / 测试 ----------
    def train(self):
        for epoch in range(1, self.args.n_epochs + 1):
            avg_train_loss = 0.0
            self.model.train()

            for i, (fea, label, cluster_ids) in enumerate(tqdm(self.train_loader)):
                self.step += 1
                fea = fea.to(self.args.device)                 # [1, N, D]
                label = label.to(self.args.device)             # [1]
                cluster_ids = cluster_ids.to(self.args.device) # [1, N]

                self.optimizer.zero_grad()

                # warmup lr
                if self.step < self.warmup_steps:
                    lr_scale = self.step / self.warmup_steps
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr_scale * self.args.lr

                loss = self.train_inference(fea, label, cluster_ids)

                if torch.isnan(loss):
                    logging.warning(f"NaN loss at step {self.step}, skip this batch")
                    continue

                avg_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            avg_train_loss /= (i + 1)
            logging.info(
                f"Epoch {epoch}, Step {self.step}, Avg Train Loss = {avg_train_loss:.4f}, "
                f"LR = {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            # 每个 epoch 在 test set 上评估一次
            test_loss, test_auc, test_acc, test_precision, test_recall, test_f1 = self.evaluate_on_test(epoch)

            if test_auc > self.best_auc:
                logging.info(
                    f"Test AUC improved from {self.best_auc:.4f} to {test_auc:.4f} at epoch {epoch}. Saving model..."
                )
                self.best_auc = test_auc
                self.best_test_metrics = [test_loss, test_auc, test_acc, test_precision, test_recall, test_f1]
                torch.save(self.model.state_dict(), self.ckpt_name)

        logging.info(
            f"Training finished. Best Test AUC = {self.best_auc:.4f}, "
            f"Metrics = {self.best_test_metrics}"
        )
        return self.best_test_metrics

    def evaluate_on_test(self, epoch=None):
        avg_loss = 0.0
        self.model.eval()
        labels, probs = [], []

        with torch.no_grad():
            for i, (fea, label, cluster_ids) in enumerate(tqdm(self.test_loader)):
                fea = fea.to(self.args.device)
                label = label.to(self.args.device)
                cluster_ids = cluster_ids.to(self.args.device)

                loss, y_prob = self.test_inference(fea, label, cluster_ids)
                labels.append(label.data.cpu().numpy())
                probs.append(y_prob.data.cpu().numpy())
                avg_loss += loss.item()

        avg_loss /= (i + 1)
        labels = np.concatenate(labels, 0)
        probs = np.concatenate(probs, 0)

        auc = self.cal_AUC(probs, labels, self.args.n_classes)
        acc, acc_log, precision, recall, f1 = self.cal_ACC(probs, labels, self.args.n_classes)

        if epoch is not None:
            logging.info(
                f"[Epoch {epoch}] Test: Loss = {avg_loss:.4f}, AUC = {auc:.4f}, Acc = {acc:.4f}, "
                f"Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}"
            )
        else:
            logging.info(
                f"Test: Loss = {avg_loss:.4f}, AUC = {auc:.4f}, Acc = {acc:.4f}, "
                f"Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}"
            )

        return avg_loss, auc, acc, precision, recall, f1

    def test(self):
        if not os.path.exists(self.ckpt_name):
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_name}")

        self.model.load_state_dict(torch.load(self.ckpt_name, map_location=self.args.device))
        self.model.eval()

        avg_loss, auc, acc, precision, recall, f1 = self.evaluate_on_test(epoch=None)
        return avg_loss, auc, acc, precision, recall, f1

    # ---------- forward 封装 ----------
    def train_inference(self, fea, label, cluster_ids):
        bag_logit = self.model(fea, cluster_ids)  # [B, n_classes]
        loss = self.loss(bag_logit, label)
        return loss

    def test_inference(self, fea, label, cluster_ids):
        bag_logit = self.model(fea, cluster_ids)
        loss = self.loss(bag_logit, label)
        y_prob = F.softmax(bag_logit, dim=1)
        return loss, y_prob

    # ---------- metrics ----------
    def cal_AUC(self, probs, labels, nclasses):
        if nclasses == 2:
            auc_score = roc_auc_score(labels, probs[:, 1])
        else:
            auc_score = roc_auc_score(labels, probs, multi_class='ovr')
        return auc_score

    def cal_ACC(self, probs, labels, nclasses):
        log = [{"count": 0, "correct": 0} for _ in range(nclasses)]
        pred_hat = np.argmax(probs, 1)
        labels = labels.astype(np.int32)

        if nclasses == 2:
            acc_score = accuracy_score(labels, pred_hat)
            precision = precision_score(labels, pred_hat, average='binary', zero_division=0)
            recall = recall_score(labels, pred_hat, average='binary', zero_division=0)
            f1 = f1_score(labels, pred_hat, average='binary', zero_division=0)
            return acc_score, log, precision, recall, f1
        else:
            acc_score = accuracy_score(labels, pred_hat)
            precision = precision_score(labels, pred_hat, average='macro', zero_division=0)
            recall = recall_score(labels, pred_hat, average='macro', zero_division=0)
            f1 = f1_score(labels, pred_hat, labels=list(range(nclasses)), average='macro', zero_division=0)
            return acc_score, log, precision, recall, f1
