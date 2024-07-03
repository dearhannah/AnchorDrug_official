# python BaseLine_finetune.py -c A549 --pretrain
# python BaseLine_finetune.py -c A549 -q KMeans --pretrain --finetune
# python BaseLine_finetune.py -c A549 -q naiveUncertainty1 --pretrain --finetune
# python BaseLine_finetune.py -c A549 -q naiveUncertainty2 --pretrain --finetune
# python BaseLine_finetune.py -c A549 --pretrain --finetune

# python BaseLine_finetune.py -c A549 --lr 0.001 --n_epoch 10 --pretrain --finetune
# python BaseLine_finetune.py -c A549 --lr 0.001 --n_epoch 20 --pretrain --finetune
# python BaseLine_finetune.py -c A549 --lr 0.001 --n_epoch 30 --pretrain --finetune
# python BaseLine_finetune.py -c A549 --lr 0.001 --n_epoch 40 --pretrain --finetune

python BaseLine_finetune.py -c A549 --n_epoch 50 --pretrain --finetune --balancesample
python BaseLine_finetune.py -c A549 --n_epoch 50 --pretrain --finetune
python BaseLine_finetune.py -c A549 --n_epoch 50 --pretrain

# python BaseLine_finetune.py -c A549 -q LeastConfidence --pretrain --finetune
# python BaseLine_finetune.py -c A549 -q MarginSampling --pretrain --finetune
# python BaseLine_finetune.py -c A549 -q KMeansSampling --pretrain --finetune
# python BaseLine_finetune.py -c A549 -q KCenterGreedy --pretrain --finetune
# python BaseLine_finetune.py -c A549 -q BadgeSampling --pretrain --finetune
# python BaseLine_finetune.py -c A549 -q BALDDropout --pretrain --finetune
# python BaseLine_finetune.py -c A549 -q RandomSampling --pretrain --finetune