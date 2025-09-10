import os
import argparse
import time

# 创建参数解析器
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--fold", type=int, default=10)
parser.add_argument("--label_frac", type=float, default=1.00)
parser.add_argument("--dataset", type=str, default='CustomDataset')
parser.add_argument("--pretrain", type=str, default='ResNet50_ImageNet')
parser.add_argument("--use_random_inst", type=str, default='False')
parser.add_argument("--init_type", type=str, default='normal')
parser.add_argument("--mask_ratio", type=float, default=0.1)
parser.add_argument("--degree", type=int, default=12)
parser.add_argument("--pretrain_dir", type=str, default='Null')
parser.add_argument("--lwc", type=str, default='Null')
parser.add_argument("--distill_loss", type=str, default='Null')
args = parser.parse_args()

# 阶段 1：训练和测试 DMIN (model=v4)
for k in range(args.fold):
    # 训练 DMIN
    cmd_train_dmin = (
        f'CUDA_VISIBLE_DEVICES={args.gpu_id} python baseline.py --phase train --dataset {args.dataset} '
        f'--pretrain {args.pretrain} --model v4 --mask_ratio {args.mask_ratio} '
        f'--fold {args.fold} --label_frac {args.label_frac} --lr {args.lr} --k {k} '
        f'--degree {args.degree} --init_type {args.init_type} --pretrain_dir {args.pretrain_dir} '
        f'--lwc {args.lwc} --distill_loss {args.distill_loss} --use_random_inst {args.use_random_inst}'
    )
    os.system(cmd_train_dmin)

    # 测试 DMIN
    cmd_test_dmin = (
        f'CUDA_VISIBLE_DEVICES={args.gpu_id} python baseline.py --phase test --dataset {args.dataset} '
        f'--pretrain {args.pretrain} --model v4 --mask_ratio {args.mask_ratio} '
        f'--fold {args.fold} --label_frac {args.label_frac} --lr {args.lr} --k {k} '
        f'--degree {args.degree} --init_type {args.init_type} --pretrain_dir {args.pretrain_dir} '
        f'--lwc {args.lwc} --distill_loss {args.distill_loss} --use_random_inst {args.use_random_inst}'
    )
    os.system(cmd_test_dmin)

# 更新 pretrain_dir 为 DMIN 的检查点路径
args.pretrain_dir = './experiments/CU10/Res50/init_normal/label_frac=1.0/model=v4_degree=12/lr=0.0002_maskratio=0.1/ckpts'

# 阶段 2：训练和测试 LIPN (model=v5)
for k in range(args.fold):
    # 训练 LIPN
    cmd_train_lipn = (
        f'CUDA_VISIBLE_DEVICES={args.gpu_id} python baseline.py --phase train --dataset {args.dataset} '
        f'--pretrain {args.pretrain} --model v5 --mask_ratio {args.mask_ratio} '
        f'--fold {args.fold} --label_frac {args.label_frac} --lr {args.lr} --k {k} '
        f'--degree {args.degree} --init_type {args.init_type} --pretrain_dir {args.pretrain_dir} '
        f'--lwc {args.lwc} --distill_loss kv --use_random_inst {args.use_random_inst}'
    )
    os.system(cmd_train_lipn)

    # 测试 LIPN
    cmd_test_lipn = (
        f'CUDA_VISIBLE_DEVICES={args.gpu_id} python baseline.py --phase test --dataset {args.dataset} '
        f'--pretrain {args.pretrain} --model v5 --mask_ratio {args.mask_ratio} '
        f'--fold {args.fold} --label_frac {args.label_frac} --lr {args.lr} --k {k} '
        f'--degree {args.degree} --init_type {args.init_type} --pretrain_dir {args.pretrain_dir} '
        f'--lwc {args.lwc} --distill_loss kv --use_random_inst {args.use_random_inst}'
    )
    os.system(cmd_test_lipn)

if __name__ == "__main__":
    print("Starting HDMIL training and testing process...")
    start_time = time.time()
    print(f"Process started at {time.strftime('%H:%M:%S', time.localtime())}")
    print(f"Using GPU ID: {args.gpu_id}")
    print(f"Learning rate: {args.lr}")
    print(f"Fold: {args.fold}")
    print(f"Completed in {time.time() - start_time:.2f} seconds")