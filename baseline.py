import argparse
import torch
import csv
import random
import os
import numpy as np
import logging

def parse_args_and_save():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--phase", type=str, default='rain')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=10)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--feature_dim", type=int, default=1536)
    parser.add_argument("--label_frac", type=float, default=1.0)
    parser.add_argument("--dataset", type=str, default="CustomDataset")
    parser.add_argument("--pretrain", type=str, default="ResNet50_ImageNet")
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--degree", type=int, default=6)
    parser.add_argument("--init_type", type=str, default='xavier')
    parser.add_argument("--use_random_inst", type=str, default='False')
    parser.add_argument("--model", type=str, default='v4')
    parser.add_argument("--mask_ratio", type=float, default=0.3)
    parser.add_argument("--pretrain_dir", type=str, default='/home/joyivan/sy/HDMIL-main/experiments/Custom10/Res50/init_xavier/label_frac=1.0/model=v4_degree=5/lr=1e-05_maskratio=0.3/ckpts/')
    parser.add_argument("--lwc", type=str, default='mbv4t')
    parser.add_argument("--distill_loss", type=str, default='l1')
    parser.add_argument("--label_1_weight", type=float, default=0.5)
    parser.add_argument("--n_classes_1", type=int, default=2)
    parser.add_argument("--cond_weight", type=float, default=0.2)
    args = parser.parse_args()
    args = init_args(args)
    return args

def init_args(args):
    args.device = torch.device('cuda')

    if args.dataset == 'Camelyon16':
        args.n_classes, args.subtyping, args.k_sample, args.n_groups = 2, False, 32, 4
        args.feature_dir = '/user_name/02.data/02.processed_data/Camelyon16/20x/feats/'
        args.label_csv = '/user_name/02.data/02.processed_data/Camelyon16_label.csv'
        args.split_dir = '/user_name/02.data/02.processed_data/Camelyon16/splits/'
    elif args.dataset == 'TCGA-NSCLC':
        args.n_classes, args.subtyping, args.k_sample = 2, True, 32
        args.feature_dir = '/user_name/02.data/02.processed_data/TCGA-NSCLC/20x/feats/'
        args.label_csv = '/user_name/02.data/02.processed_data/TCGA-NSCLC_label.csv'
        args.split_dir = '/user_name/02.data/02.processed_data/TCGA-NSCLC/splits/'
    elif args.dataset == 'TCGA-BRCA':
        args.n_classes, args.subtyping, args.k_sample, args.n_groups = 2, True, 32, 4
        args.feature_dir = '/user_name/02.data/02.processed_data/TCGA-BRCA/20x/feats/'
        args.label_csv = '/user_name/02.data/02.processed_data/TCGA-BRCA_label.csv'
        args.split_dir = '/user_name/02.data/02.processed_data/TCGA-BRCA/splits/'
    elif args.dataset == 'TCGA-RCC':
        args.n_classes, args.subtyping, args.k_sample, args.n_groups = 3, True, 8, None
        args.feature_dir = '/user_name/02.data/02.processed_data/TCGA-RCC/20x/feats/'
        args.label_csv = '/user_name/02.data/02.processed_data/TCGA-RCC_label.csv'
        args.split_dir = '/user_name/02.data/02.processed_data/TCGA-RCC/splits/'
    elif args.dataset == 'CustomDataset':
        args.n_classes, args.subtyping, args.k_sample, args.n_groups = 2, False, 16, 4
        args.feature_dir = '/media/public506/sdb/SY/TRIDENT/20x_256px_0px_overlap/features_uni_v2/'
        args.label_csv = '/media/public506/sdb/SY/labels/GenLyOs_labels.csv'
        args.split_dir = '/media/public506/sdb/SY/TRIDENT/10kfold/'
        # args.patch_dir = '/media/joyivan/2/sy/private/TREDENT/1.25x_16px_0px_overlap/patch'
    else:
        raise NotImplementedError

    return args

def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_loggers(stdout_txt):
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(stdout_txt)
    formatter = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel('INFO')

def make_expdir_and_logs(args):
    dataset = {
        'Camelyon16': 'C',
        'TCGA-NSCLC': 'N',
        'TCGA-BRCA': 'B',
        'TCGA-RCC': 'R',
        'CustomDataset': 'Custom'
    }
    pretrain = {
        'ResNet50_ImageNet': 'Res50'
    }
    if args.model == 'v4':
        exp_dir = os.path.join('./experiments/{}{}/{}/init_{}/label_frac={}/model={}_degree={}/lr={}_maskratio={}'.
                               format(dataset[args.dataset], args.fold, pretrain[args.pretrain],
                                      args.init_type,
                                      args.label_frac, args.model, args.degree, args.lr, args.mask_ratio)
                               )
    elif args.model == 'v5':
        exp_dir = os.path.join(
            './experiments/{}{}/{}/init_{}/label_frac={}/model={}_degree={}_distlosss={}/use_random_inst={}/lr={}_maskratio={}_{}'.
            format(dataset[args.dataset], args.fold, pretrain[args.pretrain],
                   args.init_type,
                   args.label_frac, args.model, args.degree, args.distill_loss, args.use_random_inst, args.lr,
                   args.mask_ratio, args.lwc)
            )
    else:
        raise NotImplementedError

    args.exp_dir = exp_dir
    args.log_dir = os.path.join(args.exp_dir, 'logs')

    if os.path.exists(args.exp_dir):
        if args.k == 0 and args.phase == 'train':
            os.system('rm -rvf {}'.format(args.exp_dir))
    else:
        os.makedirs(args.exp_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    set_loggers(os.path.join(args.log_dir, "{}-stdout-fold{}.txt".format(args.phase, args.k)))

    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp dir = {}".format(args.exp_dir))
    logging.info("Writing log file to {}".format(os.path.join(args.exp_dir, 'logs')))

def main(args):
    print(f"args.degree: {args.degree}, type: {type(args.degree)}")
    if args.phase == 'train':
        csv_name = 'valid_metrics.csv'
    elif args.phase == 'test':
        csv_name = 'test_metrics.csv'
    else:
        raise NotImplementedError

    with open(os.path.join(args.log_dir, csv_name), 'a') as f:
        writer = csv.writer(f)

        if args.k == 0:
            if args.phase == 'train':
                writer.writerow(
                    ['fold', 'valid_loss', 'valid_auc', 'valid_acc', 'valid_precision', 'valid_recall', 'valid_f1',
                     'valid_auc_1', 'valid_acc_1', 'valid_precision_1', 'valid_recall_1', 'valid_f1_1', 'score_rate'])
            else:
                writer.writerow(
                    ['fold', 'test_loss', 'test_auc', 'test_acc', 'test_precision', 'test_recall', 'test_f1',
                     'test_auc_1', 'test_acc_1', 'test_precision_1', 'test_recall_1', 'test_f1_1', 'score_rate'])

        logging.info("\n{}start fold {} {}".format(''.join(['*'] * 50), args.k, ''.join(['*'] * 50)))

        args.ckpt_dir = os.path.join(args.exp_dir, 'ckpts/fold-{}'.format(args.k))
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)

        if args.model == 'v4':
            from DMIN import DMINMIL as MIL
        elif args.model == 'v5':
            from LIPN import LIPNMIL as MIL
        else:
            raise NotImplementedError

        MIL_runner = MIL(args)

        if args.phase == 'train':
            metrics = MIL_runner.train()
        elif args.phase == 'test':
            metrics = MIL_runner.test()
        else:
            raise NotImplementedError

        writer.writerow(['{}'.format(args.k)] + [round(m, 4) for m in metrics])

if __name__ == '__main__':
    args = parse_args_and_save()
    seed_torch(args.seed)
    make_expdir_and_logs(args)
    main(args)