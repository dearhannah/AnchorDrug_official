python BaseLine_finetune.py -c A549 --pretrain
python BaseLine_finetune.py -c A549 -q KMeans --pretrain --finetune
python BaseLine_finetune.py -c A549 -q naiveUncertainty1 --pretrain --finetune
python BaseLine_finetune.py -c A549 -q naiveUncertainty2 --pretrain --finetune
# python BaseLine_finetune.py -c A549 --pretrain --finetune

# python BaseLine_finetune.py -c A549 --lr 0.001 --n_epoch 10 --pretrain --finetune
# python BaseLine_finetune.py -c A549 --lr 0.001 --n_epoch 20 --pretrain --finetune
# python BaseLine_finetune.py -c A549 --lr 0.001 --n_epoch 30 --pretrain --finetune
# python BaseLine_finetune.py -c A549 --lr 0.001 --n_epoch 40 --pretrain --finetune