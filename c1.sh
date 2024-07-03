# python BaseLine_finetune.py -c MCF7 --pretrain
# python BaseLine_finetune.py -c MCF7 -q KMeans --pretrain --finetune
# python BaseLine_finetune.py -c MCF7 -q naiveUncertainty1 --pretrain --finetune
# python BaseLine_finetune.py -c MCF7 -q naiveUncertainty2 --pretrain --finetune
# python BaseLine_finetune.py -c MCF7 --pretrain --finetune

# python BaseLine_finetune.py -c MCF7 --lr 0.001 --n_epoch 10 --pretrain --finetune
# python BaseLine_finetune.py -c MCF7 --lr 0.001 --n_epoch 20 --pretrain --finetune
# python BaseLine_finetune.py -c MCF7 --lr 0.001 --n_epoch 30 --pretrain --finetune
# python BaseLine_finetune.py -c MCF7 --lr 0.001 --n_epoch 40 --pretrain --finetune

python BaseLine_finetune.py -c MCF7 -q LeastConfidence --pretrain --finetune
python BaseLine_finetune.py -c MCF7 -q MarginSampling --pretrain --finetune
python BaseLine_finetune.py -c MCF7 -q KMeansSampling --pretrain --finetune
python BaseLine_finetune.py -c MCF7 -q KCenterGreedy --pretrain --finetune
python BaseLine_finetune.py -c MCF7 -q BadgeSampling --pretrain --finetune
python BaseLine_finetune.py -c MCF7 -q BALDDropout --pretrain --finetune
python BaseLine_finetune.py -c MCF7 -q RandomSampling --pretrain --finetune