# ===== baseline.py =====
import argparse
import csv
import logging
import os
import random
import numpy as np
import torch
from DMIN import DMINMIL


def parse_args_and_save():
    parser = argparse.ArgumentParser(description='Cluster-Embedding Hierarchical HDMIL')

    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=10)   
    parser.add_argument("--k", type=int, default=8)       
    parser.add_argument("--feature_dim", type=int, default=1536)
    parser.add_argument("--label_frac", type=float, default=1.0)
    parser.add_argument("--dataset", type=str, default="CustomDataset",
                        choices=["Camelyon16", "TCGA-NSCLC", "TCGA-BRCA", "TCGA-RCC", "CustomDataset"])
    parser.add_argument("--pretrain", type=str, default="ResNet50_ImageNet")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=1e-5)

    parser.add_argument("--label_smoothing", type=float, default=0.05,
                        help="CE 的 label smoothing 系数，0 表示关闭")
    parser.add_argument("--slide_dropout", type=float, default=0.2,
                        help="slide-level classifier 前的 Dropout 比例")
    parser.add_argument("--max_patches_per_cluster", type=int, default=300,
                        help="每个簇最多使用的 patch 数，<=0 表示不限制")

    parser.add_argument("--num_clusters", type=int, default=24,
                        help="全局聚类得到的 cluster 数 K_global")

    parser.add_argument(
        "--coord_pkl_path",
        type=str,
        default="/media/joyivan/2/sy/private/TREDENT/20x_256px_0px_overlap/outputss/global_clustering_results/coord_to_global_cluster.pkl"
    )

    args = parser.parse_args()
    args = init_args(args)
    return args


def init_args(args):
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        args.feature_dir = '/media/joyivan/2/sy/private/TREDENT/20x_256px_0px_overlap/features_uni_v2/'
        args.label_csv = '/media/joyivan/2/sy/private/labels/GenLyOs_labels.csv'
        args.split_dir = '/media/joyivan/2/sy/private/CLAM/10kfold/'
        args.coord_pkl_path = "/media/joyivan/2/sy/private/TREDENT/20x_256px_0px_overlap/outputss/global_clustering_results/coord_to_global_cluster.pkl"

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
    logger.handlers = []
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
    exp_dir = os.path.join(
        './experiments/{}{}/{}/label_frac={}/ls={}_sd={}_mppc={}_nc={}_lr={}'.format(
            dataset[args.dataset],
            args.fold,
            pretrain[args.pretrain],
            args.label_frac,
            args.label_smoothing,
            args.slide_dropout,
            args.max_patches_per_cluster,
            args.num_clusters,
            args.lr
        )
    )

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
    print(f"num_clusters: {args.num_clusters}")
    print(f"label_smoothing: {args.label_smoothing}")
    print(f"slide_dropout: {args.slide_dropout}")
    print(f"max_patches_per_cluster: {args.max_patches_per_cluster}")
    print(f"coord_pkl_path: {args.coord_pkl_path}")

    csv_name = 'test_metrics.csv'

    with open(os.path.join(args.log_dir, csv_name), 'a') as f:
        writer = csv.writer(f)
        if args.k == 0:
            if args.phase == 'train':
                writer.writerow(
                    ['fold', 'best_test_loss', 'best_test_auc', 'best_test_acc',
                     'best_test_precision', 'best_test_recall', 'best_test_f1']
                )
            else:
                writer.writerow(
                    ['fold', 'test_loss', 'test_auc', 'test_acc',
                     'test_precision', 'test_recall', 'test_f1']
                )

        logging.info("\n{}start fold {} {}".format(''.join(['*'] * 50), args.k, ''.join(['*'] * 50)))
        args.ckpt_dir = os.path.join(args.exp_dir, 'ckpts/fold-{}'.format(args.k))
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)

        MIL_runner = DMINMIL(args)

        if args.phase == 'train':
            metrics = MIL_runner.train()
            writer.writerow(['{}'.format(args.k)] + [round(m, 4) for m in metrics])
        elif args.phase == 'test':
            metrics = MIL_runner.test()
            writer.writerow(['{}'.format(args.k)] + [round(m, 4) for m in metrics])
        else:
            raise NotImplementedError


if __name__ == '__main__':
    args = parse_args_and_save()
    seed_torch(args.seed)
    make_expdir_and_logs(args)
    main(args)
