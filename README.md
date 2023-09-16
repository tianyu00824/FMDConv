# FMDConv


python -m torch.distributed.launch --nproc_per_node=4 main.py --arch od_resnet18 --epochs 100 --lr 0.1 --wd 1e-4 --lr-decay schedule --schedule 30 60 90 --kernel_num 4 --data /root/autodl-tmp/imagenet --train-batch 256

python -m torch.distributed.launch --nproc_per_node=1 main.py --arch fmd_resnet18 --epochs 100 --lr 0.1 --wd 1e-4 --lr-decay schedule --schedule 30 60 90 --kernel_num 4 --train-batch 64

python -m torch.distributed.launch --nproc_per_node=4 main.py --arch od_resnet18 --epochs 100 --lr 0.0625 --wd 1e-4 --lr-decay schedule --schedule 30 60 90 --kernel_num 4 --data /root/autodl-tmp/imagenet --train-batch 400 --dropout 0.1

python -m torch.distributed.launch --nproc_per_node=1 main.py --arch od_resnet18 --epochs 100 --lr 0.1 --wd 1e-4 --lr-decay schedule --schedule 30 60 90 --kernel_num 4 --data /root/autodl-tmp/imagenet --dropout 0.1 --train-batch 32


my Linux:
  K=2  76.57 94.10 125.20s            
  K=4  76.10 94.07 126.00s
  K=6  76.31 93.80 139.14s
  K=8  76.71 94.16 142.46s
  K=16 77.28 94.08 143.96s

  K=1  76.94 94.16 124.74s
  K=2  76.72 94.19 129.86S
  K=4  76.01 93.74 130.48s
  K=8  75.66 93.61 139.52s   76.33 93.78 135.74s 

  FMD_DCD Res18 76.64  93.94 143.59 (batch size 32)
          Res50 76.37  93.81 655.63 (batch size 16)


  FMD
  K=4 /  T = 40: 76.35 93.89 128.22s / T=52  76.51 94.30 126.72 / T=61 76.76 94.17 128.83
        T = 100: 76.34 94.21 138.04s
  Omni
  K=4  76.04 93.71 247.59s 
  
