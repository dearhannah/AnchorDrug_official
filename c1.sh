python BaseLine_finetune.py -c MCF7 --pretrain
python BaseLine_finetune.py -c MCF7 -q KMeans --pretrain --finetune
python BaseLine_finetune.py -c MCF7 -q naiveUncertainty1 --pretrain --finetune
python BaseLine_finetune.py -c MCF7 -q naiveUncertainty2 --pretrain --finetune

# python BaseLine_finetune.py -c MCF7 --lr 0.001 --n_epoch 10 --pretrain --finetune
# python BaseLine_finetune.py -c MCF7 --lr 0.001 --n_epoch 20 --pretrain --finetune
# python BaseLine_finetune.py -c MCF7 --lr 0.001 --n_epoch 30 --pretrain --finetune
# python BaseLine_finetune.py -c MCF7 --lr 0.001 --n_epoch 40 --pretrain --finetune



