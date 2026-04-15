# ===== DMIN_only_embedding.py =====
import os
import logging
import pickle

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, SequentialSampler, Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def _is_new_coord_map_format(obj) -> bool:
    # NEW: dict[wsi][(x,y)] = cluster
    if not isinstance(obj, dict):
        return False
    if len(obj) == 0:
        return True
    any_val = next(iter(obj.values()))
    return isinstance(any_val, dict)


class WSIDataset(Dataset):
    """
    WSI bag dataset with cluster ids per patch.
    Supports:
      - NEW coord map: coord_map[wsi][(x,y)] = global_cluster
      - OLD coord map: coord_map[(x,y)] = global_cluster (not recommended)
    Unknown cluster id = num_clusters
    """

    def __init__(self, args, wsi_labels, infold_cases, phase: str,
                 coord_pkl_path: str, target_cluster=None):
        self.args = args
        self.phase = phase
        self.target_cluster = target_cluster

        self.unknown_cluster_id = int(args.num_clusters)

        if not coord_pkl_path:
            raise ValueError("coord_pkl_path is required.")
        if not os.path.exists(coord_pkl_path):
            raise FileNotFoundError(f"coord pkl not found: {coord_pkl_path}")

        logging.info(f"[WSIDataset-{phase}] coord_map={coord_pkl_path}")
        with open(coord_pkl_path, "rb") as f:
            self.coord_to_cluster = pickle.load(f)

        self.coord_map_is_new = _is_new_coord_map_format(self.coord_to_cluster)
        logging.info(f"[WSIDataset-{phase}] coord_map_format={'NEW' if self.coord_map_is_new else 'OLD'}")

        self.infold_features = []
        self.infold_labels = []
        self.infold_cluster_ids = []

        for case_id, slide_id, label in wsi_labels:
            case_id = str(case_id)
            slide_id = str(slide_id)

            if case_id not in infold_cases:
                continue

            h5_path = os.path.join(args.feature_dir, f"{slide_id}.h5")
            if not os.path.exists(h5_path):
                continue

            with h5py.File(h5_path, "r") as f:
                feats = f["features"][:]  # (N, D)
                coords = f["coords"][:]   # (N, 2)

            cluster_ids = self._coords_to_cluster_ids(coords, case_id, slide_id)

            if target_cluster is not None:
                mask = (cluster_ids == int(target_cluster))
                if mask.sum() < 10:
                    continue
                feats = feats[mask]
                cluster_ids = cluster_ids[mask]

            feats_t = torch.from_numpy(feats.astype(np.float32))
            cids_t = torch.from_numpy(cluster_ids.astype(np.int64))

            if phase == "train":
                perm = torch.randperm(feats_t.shape[0])
                feats_t = feats_t[perm]
                cids_t = cids_t[perm]

            self.infold_features.append(feats_t)
            self.infold_labels.append(int(label))
            self.infold_cluster_ids.append(cids_t)

        logging.info(f"[WSIDataset-{phase}] loaded_wsis={len(self.infold_features)}")

    def _coords_to_cluster_ids(self, coords, case_id: str, slide_id: str) -> np.ndarray:
        cids = np.full((coords.shape[0],), self.unknown_cluster_id, dtype=np.int64)

        if self.coord_map_is_new:
            # 优先 slide_id，其次 case_id
            wsi_map = self.coord_to_cluster.get(slide_id) or self.coord_to_cluster.get(case_id) or {}
            for i, (x, y) in enumerate(coords):
                cids[i] = int(wsi_map.get((int(x), int(y)), self.unknown_cluster_id))
        else:
            for i, (x, y) in enumerate(coords):
                cids[i] = int(self.coord_to_cluster.get((int(x), int(y)), self.unknown_cluster_id))

        # clamp to [0, unknown]
        bad = (cids < 0) | (cids > self.unknown_cluster_id)
        cids[bad] = self.unknown_cluster_id
        return cids

    def __len__(self):
        return len(self.infold_features)

    def __getitem__(self, idx):
        feats = self.infold_features[idx]
        label = torch.tensor(self.infold_labels[idx], dtype=torch.long)
        cluster_ids = self.infold_cluster_ids[idx]
        return feats, label, cluster_ids


class DMINMIL:
    """
    Only-Embedding ablation:
      - use_cluster_emb = True
      - use_cluster_hist = False
    Model = HierarchicalMILModel (intra + inter attention)
    """

    def __init__(self, args):
        self.args = args

        self.train_loader, self.test_loader = self._init_data_wsi()

        # 强制 only-embedding（再次兜底）
        self.args.use_cluster_emb = True
        self.args.use_cluster_hist = False
        self.args.cluster_id_dropout = 0.0

        self.model = self._init_model()
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"[Model] trainable_params={total_params/1e6:.3f}M")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd)
        self.loss_fn = torch.nn.CrossEntropyLoss(
            reduction="mean",
            label_smoothing=getattr(self.args, "label_smoothing", 0.0)
        )

        self.best_auc = -1.0
        self.best_test_metrics = None
        self.ckpt_name = os.path.join(self.args.ckpt_dir, "best_epoch.pth")

        self.step = 0
        self.warmup_steps = 100

    # ---------- data ----------
    def _read_wsi_label(self):
        df = pd.read_csv(self.args.label_csv)
        wsi_labels = []
        for _, row in df.iterrows():
            if self.args.dataset == "CustomDataset":
                sid = str(row["ID"])
                wsi_labels.append([sid, sid, int(row["label"])])
            else:
                wsi_labels.append([row["case_id"], row["slide_id"], int(row["label"])])
        return wsi_labels

    def _read_in_fold_cases(self, fold_csv):
        df = pd.read_csv(fold_csv)
        train_cases, val_cases, test_cases = [], [], []
        for _, row in df.iterrows():
            if pd.notna(row.get("train", np.nan)):
                train_cases.append(str(int(row["train"])))
            if pd.notna(row.get("val", np.nan)):
                val_cases.append(str(int(row["val"])))
            if pd.notna(row.get("test", np.nan)):
                test_cases.append(str(int(row["test"])))
        return train_cases, val_cases, test_cases

    def _make_weights_for_balanced_split(self, dataset: WSIDataset):
        N = float(len(dataset))
        counts = {}
        for y in dataset.infold_labels:
            counts[y] = counts.get(y, 0) + 1
        w = [0.0] * int(N)
        for i in range(len(dataset)):
            y = dataset.infold_labels[i]
            w[i] = N / counts[y]
        return torch.DoubleTensor(w)

    def _init_data_wsi(self):
        wsi_labels = self._read_wsi_label()
        split_csv = os.path.join(self.args.split_dir, f"splits_{self.args.k}.csv")
        train_cases, val_cases, test_cases = self._read_in_fold_cases(split_csv)

        train_val_cases = list(set(train_cases + val_cases))

        coord_train = getattr(self.args, "coord_pkl_train_path",
                              getattr(self.args, "coord_pkl_path", None))
        coord_test = getattr(self.args, "coord_pkl_test_path",
                             getattr(self.args, "coord_pkl_path", None))

        train_set = WSIDataset(self.args, wsi_labels, train_val_cases, phase="train",
                               coord_pkl_path=coord_train, target_cluster=None)
        test_set = WSIDataset(self.args, wsi_labels, test_cases, phase="test",
                              coord_pkl_path=coord_test, target_cluster=None)

        if len(train_set) == 0:
            raise ValueError("Train set is empty. Check feature_dir / split cases / label_csv.")

        weights = self._make_weights_for_balanced_split(train_set)

        train_loader = DataLoader(
            train_set, batch_size=1,
            sampler=WeightedRandomSampler(weights, len(weights), replacement=True)
        )
        test_loader = DataLoader(
            test_set, batch_size=1,
            sampler=SequentialSampler(test_set)
        )
        return train_loader, test_loader

    # ---------- model ----------
    def _init_model(self):
        from models.hdmil import HierarchicalMILModel

        mppc = getattr(self.args, "max_patches_per_cluster", None)
        if mppc is not None and mppc <= 0:
            mppc = None

        model = HierarchicalMILModel(
            I=self.args.feature_dim,
            num_clusters=self.args.num_clusters,
            n_classes=self.args.n_classes,
            dropout=True,
            cluster_emb_dim=getattr(self.args, "cluster_emb_dim", 8),
            use_cluster_emb=True,
            max_patches_per_cluster=mppc,
            slide_dropout=getattr(self.args, "slide_dropout", 0.0),
        ).to(self.args.device)
        logging.info(model)
        return model

    # ---------- train/test ----------
    def train(self):
        for epoch in range(1, self.args.n_epochs + 1):
            self.model.train()
            avg_loss = 0.0

            for i, (fea, label, cluster_ids) in enumerate(tqdm(self.train_loader, desc=f"[Train] ep={epoch}")):
                self.step += 1
                fea = fea.to(self.args.device)
                label = label.to(self.args.device)
                cluster_ids = cluster_ids.to(self.args.device)

                if self.step < self.warmup_steps:
                    lr_scale = self.step / float(self.warmup_steps)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = lr_scale * self.args.lr

                self.optimizer.zero_grad(set_to_none=True)

                logits = self.model(fea, cluster_ids)
                loss = self.loss_fn(logits, label)
                if torch.isnan(loss):
                    continue

                avg_loss += float(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            avg_loss /= (i + 1)

            test_metrics = self.evaluate_on_test(epoch)
            test_loss, test_auc = test_metrics[0], test_metrics[1]

            logging.info(f"[Epoch {epoch}] train_loss={avg_loss:.4f} test_auc={test_auc:.4f}")

            if test_auc > self.best_auc:
                self.best_auc = test_auc
                self.best_test_metrics = test_metrics
                torch.save(self.model.state_dict(), self.ckpt_name)

        return self.best_test_metrics

    def test(self):
        if not os.path.exists(self.ckpt_name):
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_name}")
        self.model.load_state_dict(torch.load(self.ckpt_name, map_location=self.args.device))
        self.model.eval()
        return self.evaluate_on_test(epoch=None)

    def evaluate_on_test(self, epoch=None):
        self.model.eval()
        avg_loss = 0.0
        labels, probs = [], []

        # 测试确定性：关闭 max_patches_per_cluster 子采样
        old_mppc = getattr(self.model, "max_patches_per_cluster", None)
        self.model.max_patches_per_cluster = None

        with torch.no_grad():
            for i, (fea, label, cluster_ids) in enumerate(tqdm(self.test_loader, desc="[Test]")):
                fea = fea.to(self.args.device)
                label = label.to(self.args.device)
                cluster_ids = cluster_ids.to(self.args.device)

                logits = self.model(fea, cluster_ids)
                loss = self.loss_fn(logits, label)
                y_prob = F.softmax(logits, dim=1)

                labels.append(label.cpu().numpy())
                probs.append(y_prob.cpu().numpy())
                avg_loss += float(loss.item())

        avg_loss /= (i + 1)
        labels = np.concatenate(labels, 0)
        probs = np.concatenate(probs, 0)

        auc = self._auc(probs, labels, self.args.n_classes)
        acc, precision, recall, f1 = self._acc(probs, labels, self.args.n_classes)

        # restore
        self.model.max_patches_per_cluster = old_mppc

        return [avg_loss, auc, acc, precision, recall, f1]

    @staticmethod
    def _auc(probs, labels, nclasses):
        if nclasses == 2:
            return roc_auc_score(labels, probs[:, 1])
        return roc_auc_score(labels, probs, multi_class="ovr")

    @staticmethod
    def _acc(probs, labels, nclasses):
        pred = np.argmax(probs, 1)
        labels = labels.astype(np.int32)

        acc = accuracy_score(labels, pred)
        if nclasses == 2:
            precision = precision_score(labels, pred, average="binary", zero_division=0)
            recall = recall_score(labels, pred, average="binary", zero_division=0)
            f1 = f1_score(labels, pred, average="binary", zero_division=0)
        else:
            precision = precision_score(labels, pred, average="macro", zero_division=0)
            recall = recall_score(labels, pred, average="macro", zero_division=0)
            f1 = f1_score(labels, pred, average="macro", zero_division=0)

        return acc, precision, recall, f1