dir=./data/out/finetune_1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/countr/lib
# CUDA_VISIBLE_DEVICES=0 python -u FSC_finetune_cross.py --epochs 1000 --batch_size 8 --lr 1e-5 --output_dir ${dir} --relu_p  --title finetuning_1 --resume ./data/out/pretrain/checkpoint__pretraining_299.pth | tee logs/val_1.log
CUDA_VISIBLE_DEVICES=0 python -u FSC_test_cross\(few-shot\).py --output_dir ./data/out/results_1 --relu_p --resume ${dir}/checkpoint__finetuning_minMAE.pth --box_bound 3 | tee logs/test_1_abs.log