# FMDConv

python -m torch.distributed.launch --nproc_per_node=1 main.py --arch fmd_resnet18 --epochs 100 --lr 0.1 --wd 1e-4 --lr-decay schedule --schedule 30 60 90 --kernel_num 4 --train-batch 64
  
