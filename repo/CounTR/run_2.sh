tag=2
dir=./data/out/finetune_${tag}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/countr/lib
# CUDA_VISIBLE_DEVICES=1 python -u FSC_finetune_cross.py --wandb debug --epochs 1000 --batch_size 8 --lr 1e-5 --output_dir ${dir} --extract resnet50 --relu_p  --title finetuning_${tag} --resume ./data/out/pretrain/checkpoint__pretraining_299.pth | tee logs/val_${tag}.log
# CUDA_VISIBLE_DEVICES=1 python -u FSC_finetune_cross.py --epochs 1000 --batch_size 8 --lr 1e-5 --output_dir ${dir} --relu_p --extract resnet50 --title finetuning_${tag} --resume ./data/out/pretrain/checkpoint__pretraining_299.pth | tee logs/val_${tag}.log
CUDA_VISIBLE_DEVICES=1 python -u FSC_test_cross\(few-shot\).py --output_dir ./data/out/results_2 --relu_p --extract resnet50 --resume ${dir}/checkpoint__finetuning_minMAE.pth --box_bound 3 | tee logs/test_${tag}_abs.log