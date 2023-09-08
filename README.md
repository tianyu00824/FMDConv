# FMDConv


python -m torch.distributed.launch --nproc_per_node=4 main.py --arch od_resnet18 --epochs 100 --lr 0.1 --wd 1e-4 --lr-decay schedule --schedule 30 60 90 --kernel_num 4 --data /root/autodl-tmp/imagenet --train-batch 256

python -m torch.distributed.launch --nproc_per_node=1 main.py --arch fmd_resnet18 --epochs 100 --lr 0.1 --wd 1e-4 --lr-decay schedule --schedule 30 60 90 --kernel_num 4 --train-batch 64

python -m torch.distributed.launch --nproc_per_node=4 main.py --arch od_resnet18 --epochs 100 --lr 0.0625 --wd 1e-4 --lr-decay schedule --schedule 30 60 90 --kernel_num 4 --data /root/autodl-tmp/imagenet --train-batch 400 --dropout 0.1

python -m torch.distributed.launch --nproc_per_node=1 main.py --arch od_resnet18 --epochs 100 --lr 0.1 --wd 1e-4 --lr-decay schedule --schedule 30 60 90 --kernel_num 4 --data /root/autodl-tmp/imagenet --dropout 0.1 --train-batch 32


my Linux:
  K=4  76.88 94.04 127.96s
  K=6  
  K=8  75.97 93.49 132.56s
  K=16 77.28 94.08 143.96s
  
