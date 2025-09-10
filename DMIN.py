import torch
import pandas as pd
import os
import logging
from torch.utils.data import DataLoader, WeightedRandomSampler, SequentialSampler, Dataset
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import sys
import h5py
from collections import Counter
sys.path.append('..')

class WSIDataset(Dataset):
    def __init__(self, args, wsi_labels, infold_cases, phase=None):
        self.args = args
        self.wsi_labels = wsi_labels
        self.phase = phase
        self.infold_features, self.infold_labels, self.infold_labels_1 = [], [], []
        for case_id, slide_id, label, label_1 in wsi_labels:
            if case_id in infold_cases:
                if args.pretrain in ['ResNet50_ImageNet']:
                    if self.args.dataset in ['Camelyon16', 'TCGA-NSCLC', 'TCGA-BRCA', 'TCGA-RCC', 'CustomDataset']:
                        fea_path = os.path.join(args.feature_dir, slide_id + '.h5')
                        if os.path.exists(fea_path):
                            self.infold_features.append(fea_path)
                            self.infold_labels.append(label)
                            self.infold_labels_1.append(label_1)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
        assert len(self.infold_features) == len(self.infold_labels) == len(self.infold_labels_1), \
            f"Feature count ({len(self.infold_features)}) does not match label count ({len(self.infold_labels)}) or label_1 count ({len(self.infold_labels_1)})"
        logging.info(f"Dataset {self.phase}: {len(self.infold_features)} samples loaded. Label distribution: {Counter(self.infold_labels)}, Label_1 distribution: {Counter(self.infold_labels_1)}")

    def __len__(self):
        return len(self.infold_features)

    def __getitem__(self, idx):
        fea_path = self.infold_features[idx]
        label = self.infold_labels[idx]
        label_1 = self.infold_labels_1[idx]
        with h5py.File(fea_path, 'r') as h5_file:
            features = torch.tensor(h5_file['features'][:], dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        label_1 = torch.tensor(label_1, dtype=torch.long)
        logging.debug(f"Features shape for idx {idx}: {features.shape}, Label: {label.item()}, Label_1: {label_1.item()}")
        return features, label, label_1, fea_path

class DMINMIL:
    def __init__(self, args):
        self.args = args
        assert self.args.model == 'v4'

        if args.dataset in ['Camelyon16', 'TCGA-NSCLC', 'TCGA-BRCA', 'TCGA-RCC', 'CustomDataset']:
            self.train_loader, self.valid_loader, self.test_loader = self.init_data_wsi()
        else:
            raise NotImplementedError

        self.model = self.init_model()
        print(self.model)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params / 1e6} M")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd)

        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')
        self.loss_1 = torch.nn.CrossEntropyLoss(reduction='sum')
        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.l2_loss = torch.nn.MSELoss(reduction='mean')

        self.cond_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')

        self.counter = 0
        self.patience = 20
        self.stop_epoch = 50
        self.best_loss = float('inf')
        self.flag = 1
        self.ckpt_name = os.path.join(self.args.ckpt_dir, 'best_epoch.pth')
        self.best_valid_metrics = None
        self.step = 0
        self.warmup_steps = 100

    def read_wsi_label(self):
        data = pd.read_csv(self.args.label_csv)
        logging.info(f"Label CSV head:\n{data.head().to_string()}")
        logging.info(f"Column names: {list(data.columns)}")

        wsi_labels = []
        for i in range(len(data)):
            case_id = str(data.loc[i, "ID"]) if self.args.dataset == 'CustomDataset' else data.loc[i, "case_id"]
            slide_id = str(data.loc[i, "ID"]) if self.args.dataset == 'CustomDataset' else data.loc[i, "slide_id"]
            label = int(data.loc[i, "label"])
            label_1 = int(data.loc[i, "label.1"])

            if self.args.dataset in ['Camelyon16']:
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
                if label == 'TCGA-KICH':
                    label = 0
                elif label == 'TCGA-KIRC':
                    label = 1
                elif label == 'TCGA-KIRP':
                    label = 2
            elif self.args.dataset == 'CustomDataset':
                if label not in [0, 1]:
                    raise ValueError(f"Invalid label value {label} at row {i} in label.csv. Expected 0 or 1.")
                if label_1 not in [0, 1]:
                    raise ValueError(f"Invalid label_1 value {label_1} at row {i} in label.csv. Expected 0 or 1.")
            else:
                raise NotImplementedError

            wsi_labels.append([case_id, slide_id, label, label_1])

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
            weight[idx] = N / classes[y]

        return torch.DoubleTensor(weight)

    def init_data_wsi(self):
        wsi_labels = self.read_wsi_label()
        split_dir = os.path.join(self.args.split_dir, '')
        train_cases, valid_cases, test_cases = self.read_in_fold_cases(os.path.join(split_dir, f'splits_{self.args.k}.csv'))

        train_set = WSIDataset(self.args, wsi_labels, train_cases, 'train')
        valid_set = WSIDataset(self.args, wsi_labels, valid_cases, 'valid')
        test_set = WSIDataset(self.args, wsi_labels, test_cases, 'test')

        logging.info(f"Case/WSI number for trainset in fold-{self.args.k} = {len(train_cases)}/{len(train_set)}")
        logging.info(f"Case/WSI number for validset in fold-{self.args.k} = {len(valid_cases)}/{len(valid_set)}")
        logging.info(f"Case/WSI number for testset in fold-{self.args.k} = {len(test_cases)}/{len(test_set)}")

        if len(train_set) == 0:
            raise ValueError("Train set is empty. Check case_id matching or file paths.")

        weights = self.make_weights_for_balanced_classes_split(train_set)
        train_loader = DataLoader(
            train_set, batch_size=1, sampler=WeightedRandomSampler(weights, len(weights), replacement=True)
        )
        valid_loader = DataLoader(valid_set, batch_size=1, sampler=SequentialSampler(valid_set))
        test_loader = DataLoader(test_set, batch_size=1, sampler=SequentialSampler(test_set))

        return train_loader, valid_loader, test_loader

    def init_model(self):
        from models.hdmil import KAN_CLAM_MB_v4, SmoothTop1SVM
        model = KAN_CLAM_MB_v4(
            I=self.args.feature_dim,
            dropout=True,
            n_classes=self.args.n_classes,
            n_classes_1=self.args.n_classes_1,
            subtyping=self.args.subtyping,
            instance_loss_fn=SmoothTop1SVM(n_classes=2).cuda(self.args.device),
            k_sample=self.args.k_sample,
            args=self.args
        ).to(self.args.device)
        return model

    def train(self):
        for epoch in range(1, self.args.n_epochs + 1):
            avg_train_loss = 0
            self.model.train()

            for i, (fea, label, label_1, fea_path) in enumerate(tqdm(self.train_loader)):
                self.step += 1
                fea, label, label_1 = fea.to(self.args.device), label.to(self.args.device), label_1.to(self.args.device)
                self.optimizer.zero_grad()

                if self.step < self.warmup_steps:
                    lr_scale = self.step / self.warmup_steps
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr_scale * self.args.lr

                loss = self.train_inference(fea, label, label_1)

                if torch.isnan(loss):
                    logging.warning(f"NaN loss detected at step {self.step}, skipping")
                    continue

                avg_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            avg_train_loss /= (i + 1)
            logging.info(f"Epoch {epoch}, Step {self.step}, Avg Train Loss = {avg_train_loss:.4f}, LR = {self.optimizer.param_groups[0]['lr']:.6f}")
            self.valid(epoch)

            if self.flag == -1:
                break

        return self.best_valid_metrics

    def valid(self, epoch):
        avg_loss = 0
        self.model.eval()
        labels, probs, labels_1, probs_1 = [], [], [], []

        for i, (fea, label, label_1, _) in enumerate(tqdm(self.valid_loader)):
            fea, label, label_1 = fea.to(self.args.device), label.to(self.args.device), label_1.to(self.args.device)
            with torch.no_grad():
                loss, y_prob, y_prob_1 = self.test_inference(fea, label, label_1)
            labels.append(label.data.cpu().numpy())
            probs.append(y_prob.data.cpu().numpy())
            labels_1.append(label_1.data.cpu().numpy())
            probs_1.append(y_prob_1.data.cpu().numpy())
            avg_loss += loss.item()

        avg_loss /= (i + 1)
        labels, probs = np.concatenate(labels, 0), np.concatenate(probs, 0)
        labels_1, probs_1 = np.concatenate(labels_1, 0), np.concatenate(probs_1, 0)

        auc = self.cal_AUC(probs, labels, self.args.n_classes)
        acc, acc_log, precision, recall, f1 = self.cal_ACC(probs, labels, self.args.n_classes)
        auc_1 = self.cal_AUC(probs_1, labels_1, self.args.n_classes_1)
        acc_1, acc_log_1, precision_1, recall_1, f1_1 = self.cal_ACC(probs_1, labels_1, self.args.n_classes_1)

        logging.info(f"Valid Epoch {epoch}: Loss = {avg_loss:.4f}, AUC = {auc:.4f}, Acc = {acc:.4f}, "
                     f"Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")
        logging.info(f"Valid Epoch {epoch} (Label_1): AUC = {auc_1:.4f}, Acc = {acc_1:.4f}, "
                     f"Precision = {precision_1:.4f}, Recall = {recall_1:.4f}, F1 = {f1_1:.4f}")
        logging.info(f"Sample labels: {labels[:5]}, Sample probs (positive class): {probs[:5, 1]}")
        logging.info(f"Sample labels_1: {labels_1[:5]}, Sample probs_1 (positive class): {probs_1[:5, 1]}")

        if epoch >= self.stop_epoch:
            if avg_loss < self.best_loss:
                self.counter = 0
                logging.info(f'Validation loss decreased ({self.best_loss:.4f} --> {avg_loss:.4f}). Saving model ...')
                torch.save(self.model.state_dict(), self.ckpt_name)
                self.best_loss = avg_loss
                self.best_valid_metrics = [avg_loss, auc, acc, precision, recall, f1, auc_1, acc_1, precision_1, recall_1, f1_1]
            else:
                self.counter += 1
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.flag = -1

    def test(self):
        avg_loss = 0
        self.model.load_state_dict(torch.load(self.ckpt_name))
        self.model.eval()
        labels, probs, labels_1, probs_1 = [], [], [], []

        for i, (fea, label, label_1, _) in enumerate(tqdm(self.test_loader)):
            fea, label, label_1 = fea.to(self.args.device), label.to(self.args.device), label_1.to(self.args.device)
            with torch.no_grad():
                loss, y_prob, y_prob_1 = self.test_inference(fea, label, label_1)
            labels.append(label.data.cpu().numpy())
            probs.append(y_prob.data.cpu().numpy())
            labels_1.append(label_1.data.cpu().numpy())
            probs_1.append(y_prob_1.data.cpu().numpy())
            avg_loss += loss.item()

        avg_loss /= (i + 1)
        labels, probs = np.concatenate(labels, 0), np.concatenate(probs, 0)
        labels_1, probs_1 = np.concatenate(labels_1, 0), np.concatenate(probs_1, 0)

        auc = self.cal_AUC(probs, labels, self.args.n_classes)
        acc, acc_log, precision, recall, f1 = self.cal_ACC(probs, labels, self.args.n_classes)
        auc_1 = self.cal_AUC(probs_1, labels_1, self.args.n_classes_1)
        acc_1, acc_log_1, precision_1, recall_1, f1_1 = self.cal_ACC(probs_1, labels_1, self.args.n_classes_1)

        logging.info(f"Test: Loss = {avg_loss:.4f}, AUC = {auc:.4f}, Acc = {acc:.4f}, "
                     f"Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")
        logging.info(f"Test (Label_1): AUC = {auc_1:.4f}, Acc = {acc_1:.4f}, "
                     f"Precision = {precision_1:.4f}, Recall = {recall_1:.4f}, F1 = {f1_1:.4f}")
        logging.info(f"Test labels: {labels[:5]}, Test probs (positive class): {probs[:5, 1]}")
        logging.info(f"Test labels_1: {labels_1[:5]}, Test probs_1 (positive class): {probs_1[:5, 1]}")

        return avg_loss, auc, acc, precision, recall, f1, auc_1, acc_1, precision_1, recall_1, f1_1

    def train_inference(self, fea, label, label_1):
        bag_logit, _, results_dict, M, select_logit, select_M, mask_rate, bag_logit_1, _, results_dict_1, M_1, select_logit_1, select_M_1, mask_rate_1 = self.model(fea, label=label, label_1=label_1, instance_eval=True)
        cls_loss = 0.7 * self.loss(bag_logit, label) + 0.3 * results_dict['instance_loss'].mean()
        cls_loss_1 = 0.7 * self.loss_1(bag_logit_1, label_1) + 0.3 * results_dict_1['instance_loss_1'].mean()
        kl_loss = self.kl_loss(F.log_softmax(select_logit, dim=1), F.log_softmax(bag_logit, dim=1))
        dis_loss = self.l2_loss(select_M, M)
        rate_loss = self.l2_loss(mask_rate, torch.tensor(self.args.mask_ratio, device=self.args.device))
        kl_loss_1 = self.kl_loss(F.log_softmax(select_logit_1, dim=1), F.log_softmax(bag_logit_1, dim=1))
        dis_loss_1 = self.l2_loss(select_M_1, M_1)
        rate_loss_1 = self.l2_loss(mask_rate_1, torch.tensor(self.args.mask_ratio, device=self.args.device))

        cond_loss = torch.tensor(0.0, device=self.args.device)
        if label.item() == 1:
            target = torch.tensor([1.0], device=self.args.device)
            cond_loss = self.cond_loss_fn(bag_logit_1[:, 1], target)

        loss = (cls_loss + 0.5 * kl_loss + 0.5 * dis_loss + 1.0 * rate_loss) + \
               self.args.label_1_weight * (cls_loss_1 + 0.5 * kl_loss_1 + 0.5 * dis_loss_1 + 1.0 * rate_loss_1) + \
               self.args.cond_weight * cond_loss

        logging.info(f"Step {self.step}: cls_loss={cls_loss.item():.4f}, cls_loss_1={cls_loss_1.item():.4f}, "
                     f"kl_loss={kl_loss.item():.4f}, dis_loss={dis_loss.item():.4f}, rate_loss={rate_loss.item():.4f}, "
                     f"kl_loss_1={kl_loss_1.item():.4f}, dis_loss_1={dis_loss_1.item():.4f}, rate_loss_1={rate_loss_1.item():.4f}, "
                     f"cond_loss={cond_loss.item():.4f}, "
                     f"mask_rate={mask_rate.item():.4f}, mask_rate_1={mask_rate_1.item():.4f}")
        logging.debug(f"Bag logit: {bag_logit[0].detach().cpu().numpy()}, Select logit: {select_logit[0].detach().cpu().numpy()}")
        logging.debug(f"Bag logit_1: {bag_logit_1[0].detach().cpu().numpy()}, Select logit_1: {select_logit_1[0].detach().cpu().numpy()}")
        return loss

    def test_inference(self, fea, label, label_1):
        bag_logit, bag_logit_1 = self.model(fea)
        loss = self.loss(bag_logit, label) + self.args.label_1_weight * self.loss_1(bag_logit_1, label_1)
        y_prob = F.softmax(bag_logit, dim=1)
        y_prob_1 = F.softmax(bag_logit_1, dim=1)
        return loss, y_prob, y_prob_1

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