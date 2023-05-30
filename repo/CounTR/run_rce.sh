# prolonged tag=3
tag=rce
dir=./data/out/finetune_${tag}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/countr/lib
# CUDA_VISIBLE_DEVICES=1 python -u FSC_finetune_cross.py --wandb debug --epochs 1000 --batch_size 8 --lr 1e-5 --output_dir ${dir} --extract resnet50 --relu_p  --title finetuning_${tag} --resume ./data/out/pretrain/checkpoint__pretraining_299.pth | tee logs/val_${tag}.log
# CUDA_VISIBLE_DEVICES=2 python -u FSC_finetune_cross.py --wandb counting --epochs 500 --loss_fn rce --lr_sched --epochs_per_save 10 --batch_size 8 --lr 1e-5 --output_dir ${dir} --adaption --extract resnet18 --title finetuning_${tag} --resume ./data/out/pretrain/checkpoint__pretraining_299.pth | tee logs/val_${tag}.log
CUDA_VISIBLE_DEVICES=1 python -u FSC_test_cross\(few-shot\).py --output_dir ./data/out/results_${tag} --extract resnet18 --adaption --resume ${dir}/checkpoint__finetuning_minRMSE.pth --box_bound 3 | tee logs/finetune_${tag}_mulscale.log
