# train VGG16Trans model on FSC
python train.py --tag fsc-baseline --no-wandb --device 0 --scheduler step --step 40 --dcsize 8 \
--batch-size 8 --max-epoch 100 --lr 4e-4 --val-start 0 --val-epoch 1
