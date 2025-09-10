import h5py
import torch
import pandas as pd
import os
import itertools
import logging
import json
import itertools
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler, Dataset
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import sys
from collections import Counter
sys.path.append('..')
from tqdm import tqdm
import random

class DualWSIDataset(Dataset):
    def __init__(self, args, wsi_labels, infold_cases, phase=None):
        self.args = args
        self.wsi_labels = wsi_labels
        self.phase = phase
        self.low_level_dir = '/media/joyivan/2/sy/private/CLAM/patch5*/extracted_patches/'
        self.infold_features, self.infold_labels = [], []

        for case_id, slide_id, label in wsi_labels:
            if case_id in infold_cases:
                if args.pretrain in ['ResNet50_ImageNet']:
                    if self.args.dataset in ['CustomDataset']:
                        fea_path = os.path.join(args.feature_dir, slide_id + '.pt')
                        img_path = os.path.join(self.low_level_dir, slide_id + '_patches.h5')
                        if not os.path.exists(fea_path):
                            logging.warning(f"Feature file not found: {fea_path}")
                            continue
                        if not os.path.exists(img_path):
                            logging.warning(f"Image file not found: {img_path}")
                            continue

                        # 加载并验证特征
                        try:
                            features = torch.load(fea_path, weights_only=False)
                            if not isinstance(features, torch.Tensor):
                                logging.warning(f"Invalid feature type at {fea_path}, expected torch.Tensor")
                                continue
                            nfeatures = features.shape[0]
                        except Exception as e:
                            logging.warning(f"Error loading features from {fea_path}: {e}")
                            continue

                        # 加载 HDF5 文件中的 patch
                        try:
                            with h5py.File(img_path, 'r') as f:
                                if 'patches' not in f:
                                    logging.warning(f"No 'patches' dataset in {img_path}")
                                    continue
                                patches = f['patches'][:]
                                npatches = patches.shape[0]
                            if nfeatures != npatches:
                                logging.info(
                                    f"Skipping slide {slide_id}: nfeatures={nfeatures}, npatches={npatches}, difference={nfeatures - npatches}"
                                )
                                continue
                        except Exception as e:
                            logging.warning(f"Error loading patches from {img_path}: {e}")
                            continue

                        # 如果特征和 patch 数量匹配，添加到列表
                        self.infold_features.append([fea_path, img_path])
                        self.infold_labels.append(label)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

        if len(self.infold_features) == 0:
            raise ValueError(
                f"No valid samples loaded for {self.phase} set. Check file paths and feature/patch consistency.")

        assert len(self.infold_features) == len(self.infold_labels), \
            f"Feature count ({len(self.infold_features)}) does not match label count ({len(self.infold_labels)})"
        logging.info(
            f"Dataset {self.phase}: {len(self.infold_features)} samples loaded. Label distribution: {Counter(self.infold_labels)}")

    def __len__(self):
        return len(self.infold_features)

    def __getitem__(self, index):
        fea_path, img_path = self.infold_features[index]
        label = self.infold_labels[index]
        fea = torch.load(fea_path, weights_only=False)
        with h5py.File(img_path, 'r') as f:
            img = f['patches'][:]
        img = torch.tensor(img, dtype=torch.float32)[:, :, :, :3]
        assert fea.shape[0] == img.shape[0], \
            f"Feature patches ({fea.shape[0]}) != Image patches ({img.shape[0]}) for {fea_path}"
        # Image preprocessing
        img = (img - torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3)
        logging.debug(f"Index {index}: Feature shape {fea.shape}, Image shape {img.shape}, Label {label}")
        return fea, label, fea_path, img, img_path

class LIPNMIL:
    def __init__(self, args):
        self.args = args
        assert self.args.model == 'v5'

        if args.dataset in ['Camelyon16', 'TCGA-NSCLC', 'TCGA-BRCA', 'TCGA-RCC', 'CustomDataset']:
            self.train_loader, self.valid_loader, self.test_loader = self.init_data_wsi()
        else:
            raise NotImplementedError

        self.model, self.img_processor = self.init_model()
        print(self.model)
        print(self.img_processor)

        for p in self.model.parameters():
            p.requires_grad = False

        pretrain_path = os.path.join(self.args.pretrain_dir, 'fold-{}/best_epoch.pth'.format(args.k))
        print(self.model.load_state_dict(torch.load(pretrain_path)))

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) + sum(
            p.numel() for p in self.img_processor.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params / 1e6} M")

        self.optimizer = torch.optim.Adam(self.img_processor.parameters(), lr=args.lr, weight_decay=args.wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.n_epochs, eta_min=1e-6)

        class_weights = torch.tensor([1.0, 1.5]).to(self.args.device)  # Positive class weight
        self.loss = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.l2_loss = torch.nn.MSELoss(reduction='mean')
        self.l1_loss = torch.nn.L1Loss(reduction='mean')

        self.counter = 0
        self.patience = 20
        self.stop_epoch = 50  # Align with DMIN
        self.best_loss = float('inf')
        self.flag = 1
        self.ckpt_name1 = os.path.join(self.args.ckpt_dir, 'best_mil.pth')
        self.ckpt_name2 = os.path.join(self.args.ckpt_dir, 'best_processor.pth')
        self.best_valid_metrics = None
        self.avg_mask_rate = 0.0
        self.avg_score_rate = 0.0
        self.step = 0
        self.warmup_steps = 100  # Add warmup from DMIN

    def read_wsi_label(self):
        data = pd.read_csv(self.args.label_csv)
        logging.info(f"Label CSV head:\n{data.head().to_string()}")
        logging.info(f"Column names: {list(data.columns)}")

        wsi_labels = []
        for i in range(len(data)):
            if self.args.dataset == 'CustomDataset':
                if 'ID' not in data.columns or 'label' not in data.columns:
                    raise ValueError(f"Expected columns 'ID' and 'label' in {self.args.label_csv}, got {list(data.columns)}")
                case_id = str(data.loc[i, "ID"])
                slide_id = str(data.loc[i, "ID"])
                label = int(data.loc[i, "label"])
                if label not in [0, 1]:
                    raise ValueError(f"Invalid label {label} at row {i} in {self.args.label_csv}. Expected 0 or 1.")
            else:
                case_id = str(data.loc[i, "case_id"])
                slide_id = str(data.loc[i, "slide_id"])
                label = data.loc[i, "label"]
                if self.args.dataset == 'Camelyon16':
                    assert label in ['tumor_tissue', 'normal_tissue'], f"Invalid label {label} at row {i}"
                    label = 0 if label == 'normal_tissue' else 1
                elif self.args.dataset == 'TCGA-NSCLC':
                    assert label in ['TCGA-LUSC', 'TCGA-LUAD'], f"Invalid label {label} at row {i}"
                    label = 0 if label == 'TCGA-LUSC' else 1
                elif self.args.dataset == 'TCGA-BRCA':
                    assert label in ['Infiltrating Ductal Carcinoma', 'Infiltrating Lobular Carcinoma'], f"Invalid label {label} at row {i}"
                    label = 0 if label == 'Infiltrating Ductal Carcinoma' else 1
                elif self.args.dataset == 'TCGA-RCC':
                    assert label in ['TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP'], f"Invalid label {label} at row {i}"
                    label = 0 if label == 'TCGA-KICH' else 1 if label == 'TCGA-KIRC' else 2
                else:
                    raise NotImplementedError

            wsi_labels.append([case_id, slide_id, label])

        logging.info(f"Loaded {len(wsi_labels)} labels, sample: {wsi_labels[:5]}")
        return wsi_labels

    def read_in_fold_cases(self, fold_csv):
        data = pd.read_csv(fold_csv)
        train_cases, valid_cases, test_cases = [], [], []
        for i in range(len(data)):
            train_val = data.loc[i, 'train']
            if pd.notna(train_val):
                train_cases.append(str(int(train_val)))
            else:
                logging.warning(f"NaN found in train column at row {i}, skipping")
            val_val = data.loc[i, 'val']
            if pd.notna(val_val):
                valid_cases.append(str(int(val_val)))
            else:
                logging.warning(f"NaN found in val column at row {i}, skipping")
            test_val = data.loc[i, 'test']
            if pd.notna(test_val):
                test_cases.append(str(int(test_val)))
            else:
                logging.warning(f"NaN found in test column at row {i}, skipping")

        logging.info(f"Train cases: {len(train_cases)}, sample: {train_cases[:5]}")
        logging.info(f"Valid cases: {len(valid_cases)}, sample: {valid_cases[:5]}")
        logging.info(f"Test cases: {len(test_cases)}, sample: {test_cases[:5]}")
        return train_cases, valid_cases, test_cases

    def make_weights_for_balanced_classes_split(self, data_set):
        N = float(len(data_set))
        classes = {}
        for label in data_set.infold_labels:
            if label not in classes:
                classes[label] = 1
            else:
                classes[label] += 1
        weight = [0] * int(N)
        for idx in range(len(data_set)):
            y = data_set.infold_labels[idx]
            weight[idx] = N / classes[y] * (1.5 if y == 1 else 1.0)  # Positive class weight boost
        return torch.DoubleTensor(weight)

    def init_data_wsi(self):
        wsi_labels = self.read_wsi_label()
        split_dir = os.path.join(self.args.split_dir, '')
        fold_csv = os.path.join(split_dir, f'splits_{self.args.k}.csv')
        if not os.path.exists(fold_csv):
            raise FileNotFoundError(f"Fold CSV not found: {fold_csv}")
        train_cases, valid_cases, test_cases = self.read_in_fold_cases(fold_csv)

        train_set = DualWSIDataset(self.args, wsi_labels, train_cases, 'train')
        valid_set = DualWSIDataset(self.args, wsi_labels, valid_cases, 'valid')
        test_set = DualWSIDataset(self.args, wsi_labels, test_cases, 'test')

        logging.info(f"Case/WSI number for trainset in fold-{self.args.k} = {len(train_cases)}/{len(train_set)}")
        logging.info(f"Case/WSI number for validset in fold-{self.args.k} = {len(valid_cases)}/{len(valid_set)}")
        logging.info(f"Case/WSI number for testset in fold-{self.args.k} = {len(test_cases)}/{len(test_set)}")

        if len(train_set) == 0:
            raise ValueError("Train set is empty. Check case_id matching or file paths in feature_dir and low_level_dir.")

        weights = self.make_weights_for_balanced_classes_split(train_set)
        train_loader = DataLoader(
            train_set, batch_size=1, sampler=WeightedRandomSampler(weights, len(weights), replacement=True)
        )
        valid_loader = DataLoader(valid_set, batch_size=1, sampler=SequentialSampler(valid_set))
        test_loader = DataLoader(test_set, batch_size=1, sampler=SequentialSampler(test_set))

        return train_loader, valid_loader, test_loader

    def init_model(self):
        from models.hdmil import KAN_CLAM_MB_v5, SmoothTop1SVM, ImageProcessor
        model = KAN_CLAM_MB_v5(
            I=self.args.feature_dim, dropout=True, n_classes=self.args.n_classes, subtyping=self.args.subtyping,
            instance_loss_fn=SmoothTop1SVM(n_classes=2).cuda(self.args.device), k_sample=self.args.k_sample, args=self.args
        ).to(self.args.device)
        img_processor = ImageProcessor(args=self.args).to(self.args.device)
        return model, img_processor

    def train(self):
        for epoch in range(1, self.args.n_epochs + 1):
            avg_train_loss, avg_mask_rate, avg_score_rate = 0, 0, 0
            self.model.eval()
            self.img_processor.train()

            for i, (fea, label, fea_path, img, img_path) in enumerate(tqdm(self.train_loader)):
                self.step += 1
                fea, label, img = fea.to(self.args.device), label.to(self.args.device), img.to(self.args.device)
                self.optimizer.zero_grad()

                # Learning rate warmup
                if self.step < self.warmup_steps:
                    lr_scale = self.step / self.warmup_steps
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr_scale * self.args.lr

                loss, mask_rate, score_rate = self.train_inference(fea, label, img)
                if torch.isnan(loss):
                    logging.warning(f"NaN loss at step {self.step}, skipping")
                    continue

                avg_train_loss += loss.item()
                avg_mask_rate += mask_rate.item()
                avg_score_rate += score_rate.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.img_processor.parameters(), max_norm=1.0)  # Add gradient clipping
                self.optimizer.step()
                self.scheduler.step()

            avg_train_loss /= (i + 1)
            avg_mask_rate /= (i + 1)
            avg_score_rate /= (i + 1)
            self.avg_mask_rate = avg_mask_rate
            self.avg_score_rate = avg_score_rate

            logging.info(f"Step {self.step} (epoch {epoch}), avg train loss = {avg_train_loss:.4f}, "
                         f"mask rate = {avg_mask_rate:.4f}, score rate = {avg_score_rate:.4f}, "
                         f"LR = {self.optimizer.param_groups[0]['lr']:.6f}")

            self.valid(epoch)
            if self.flag == -1:
                break

        return self.best_valid_metrics

    def valid(self, epoch):
        avg_loss = 0
        self.model.eval()
        self.img_processor.eval()
        labels, probs = [], []

        for i, (fea, label, fea_path, img, img_path) in enumerate(tqdm(self.valid_loader)):
            fea, label, img = fea.to(self.args.device), label.to(self.args.device), img.to(self.args.device)
            with torch.no_grad():
                loss, y_prob = self.test_inference(fea, label, img)
            labels.append(label.data.cpu().numpy())
            probs.append(y_prob.data.cpu().numpy())
            avg_loss += loss.item()

        avg_loss /= (i + 1)
        labels, probs = np.concatenate(labels, 0), np.concatenate(probs, 0)
        auc = self.cal_AUC(probs, labels, self.args.n_classes)
        acc, acc_log, precision, recall, f1 = self.cal_ACC(probs, labels, self.args.n_classes)

        logging.info(f"Valid Epoch {epoch}: Loss = {avg_loss:.4f}, AUC = {auc:.4f}, Acc = {acc:.4f}, "
                     f"Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")
        logging.info(f"Sample labels: {labels[:5]}, Sample probs (positive): {probs[:5, 1]}")
        logging.info(f"Probs mean: {probs[:, 1].mean():.4f}, std: {probs[:, 1].std():.4f}")

        if epoch >= self.stop_epoch:
            if avg_loss < self.best_loss:
                self.counter = 0
                logging.info(f'Validation loss decreased ({self.best_loss:.4f} --> {avg_loss:.4f}). Saving model ...')
                torch.save(self.model.state_dict(), self.ckpt_name1)
                torch.save(self.img_processor.state_dict(), self.ckpt_name2)
                self.best_loss = avg_loss
                self.best_valid_metrics = [avg_loss, auc, acc, precision, recall, f1]
            else:
                self.counter += 1
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.flag = -1

    def test(self):
        avg_loss = 0
        self.model.load_state_dict(torch.load(self.ckpt_name1))
        self.img_processor.load_state_dict(torch.load(self.ckpt_name2))
        self.model.eval()
        self.img_processor.eval()
        labels, probs = [], []

        for i, (fea, label, fea_path, img, img_path) in enumerate(tqdm(self.test_loader)):
            fea, label, img = fea.to(self.args.device), label.to(self.args.device), img.to(self.args.device)
            with torch.no_grad():
                loss, y_prob = self.test_inference(fea, label, img)
            labels.append(label.data.cpu().numpy())
            probs.append(y_prob.data.cpu().numpy())
            avg_loss += loss.item()

        avg_loss /= (i + 1)
        labels, probs = np.concatenate(labels, 0), np.concatenate(probs, 0)
        auc = self.cal_AUC(probs, labels, self.args.n_classes)
        acc, acc_log, precision, recall, f1 = self.cal_ACC(probs, labels, self.args.n_classes)

        logging.info(f"Test: Loss = {avg_loss:.4f}, AUC = {auc:.4f}, Acc = {acc:.4f}, "
                     f"Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")
        logging.info(f"Test labels: {labels[:5]}, Test probs (positive): {probs[:5, 1]}")
        logging.info(f"Test probs mean: {probs[:, 1].mean():.4f}, std: {probs[:, 1].std():.4f}")

        return avg_loss, auc, acc, precision, recall, f1

    def train_inference(self, fea, label, img=None):
        assert self.args.use_random_inst == 'False'
        with torch.no_grad():
            _, mask = self.model(fea)
        mask_add = mask[:, :, 0] + mask[:, :, 1]
        mask_rate = (mask_add != 0).sum() / (mask.shape[0] * mask.shape[1])

        img_scores = self.img_processor(img)
        score_add = img_scores[:, :, 0] + img_scores[:, :, 1]
        score_union = torch.zeros_like(score_add, memory_format=torch.legacy_contiguous_format).masked_fill(score_add != 0, 1.0)
        score_rate = (score_union - score_add.detach() + score_add).sum() / (mask.shape[0] * mask.shape[1])

        if self.args.distill_loss == 'l1':
            loss = self.l1_loss(img_scores, mask) + 0.5 * self.l2_loss(score_rate, mask_rate.detach())
        elif self.args.distill_loss == 'kv':
            loss = self.kl_loss(F.log_softmax(img_scores[0]), F.log_softmax(mask[0]))
        else:
            raise NotImplementedError

        logging.info(f"Step {self.step}: distill_loss={loss.item():.4f}, mask_rate={mask_rate.item():.4f}, score_rate={score_rate.item():.4f}")
        return loss, mask_rate, score_rate

    def test_inference(self, fea, label, img=None):
        img_scores = self.img_processor(img)
        bag_logit = self.model(fea, img_scores)[0]
        loss = self.loss(bag_logit, label)
        y_prob = F.softmax(bag_logit, dim=1)
        return loss, y_prob

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
            f1 = f1_score(labels, pred_hat, labels=list(range(self.args.n_classes)), average='macro', zero_division=0)
            return acc_score, log, precision, recall, f1