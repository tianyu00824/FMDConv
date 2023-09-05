# FMDConv


python -m torch.distributed.launch --nproc_per_node=4 main.py --arch od_resnet18 --epochs 100 --lr 0.1 --wd 1e-4 --lr-decay schedule --schedule 30 60 90 --kernel_num 4 --data /root/autodl-tmp/imagenet --train-batch 256

python -m torch.distributed.launch --nproc_per_node=1 main.py --arch fmd_resnet18 --epochs 100 --lr 0.1 --wd 1e-4 --lr-decay schedule --schedule 30 60 90 --kernel_num 4 --train-batch 64

python -m torch.distributed.launch --nproc_per_node=4 main.py --arch od_resnet18 --epochs 100 --lr 1/16 --wd 1e-4 --lr-decay schedule --schedule 30 60 90 --kernel_num 4 --data /root/autodl-tmp/imagenet --train-batch 256
