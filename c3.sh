python BaseLine_finetune.py -c PC3 --pretrain
python BaseLine_finetune.py -c PC3 -q KMeans --pretrain --finetune
python BaseLine_finetune.py -c PC3 -q naiveUncertainty1 --pretrain --finetune
python BaseLine_finetune.py -c PC3 -q naiveUncertainty2 --pretrain --finetune
# python BaseLine_finetune.py -c PC3 --pretrain --finetune

# python BaseLine_finetune.py -c PC3 --lr 0.001 --n_epoch 10 --pretrain --finetune
# python BaseLine_finetune.py -c PC3 --lr 0.001 --n_epoch 20 --pretrain --finetune
# python BaseLine_finetune.py -c PC3 --lr 0.001 --n_epoch 30 --pretrain --finetune
# python BaseLine_finetune.py -c PC3 --lr 0.001 --n_epoch 40 --pretrain --finetune