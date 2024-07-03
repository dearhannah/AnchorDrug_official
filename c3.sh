# python BaseLine_finetune.py -c PC3 --pretrain
# python BaseLine_finetune.py -c PC3 -q KMeans --pretrain --finetune
# python BaseLine_finetune.py -c PC3 -q naiveUncertainty1 --pretrain --finetune
# python BaseLine_finetune.py -c PC3 -q naiveUncertainty2 --pretrain --finetune
# python BaseLine_finetune.py -c PC3 --pretrain --finetune

# python BaseLine_finetune.py -c PC3 --lr 0.001 --n_epoch 10 --pretrain --finetune
# python BaseLine_finetune.py -c PC3 --lr 0.001 --n_epoch 20 --pretrain --finetune
# python BaseLine_finetune.py -c PC3 --lr 0.001 --n_epoch 30 --pretrain --finetune
# python BaseLine_finetune.py -c PC3 --lr 0.001 --n_epoch 40 --pretrain --finetune


# python BaseLine_finetune.py -c PC3 --n_epoch 50 --pretrain --finetune --balancesample
# python BaseLine_finetune.py -c MCF7 --n_epoch 50 --pretrain --finetune --balancesample
# python BaseLine_finetune.py -c A549 --n_epoch 50 --pretrain --finetune --balancesample
# python BaseLine_finetune.py -c PC3 --n_epoch 50 --pretrain --finetune
# python BaseLine_finetune.py -c MCF7 --n_epoch 50 --pretrain --finetune
# python BaseLine_finetune.py -c A549 --n_epoch 50 --pretrain --finetune

# python BaseLine_finetune.py -c PC3 --n_epoch 50 --pretrain
# python BaseLine_finetune.py -c MCF7 --n_epoch 50 --pretrain
# python BaseLine_finetune.py -c A549 --n_epoch 50 --pretrain

python BaseLine_finetune.py -c PC3 -q LeastConfidence --pretrain --finetune
python BaseLine_finetune.py -c PC3 -q MarginSampling --pretrain --finetune
python BaseLine_finetune.py -c PC3 -q KMeansSampling --pretrain --finetune
python BaseLine_finetune.py -c PC3 -q KCenterGreedy --pretrain --finetune
python BaseLine_finetune.py -c PC3 -q BadgeSampling --pretrain --finetune
python BaseLine_finetune.py -c PC3 -q BALDDropout --pretrain --finetune
python BaseLine_finetune.py -c PC3 -q RandomSampling --pretrain --finetune