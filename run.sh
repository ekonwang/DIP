# train VGG16Trans model on FSC
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/countr/lib
python train.py --tag fsc-baseline --no-wandb --device 1 --scheduler step --step 40 --dcsize 8 \
--batch-size 8 --max-epoch 1000 --lr 5e-5 --val-start 0 --val-epoch 1 --resume /workspace/DIP/checkpoint/0519_fsc-baseline/99_ckpt.tar
