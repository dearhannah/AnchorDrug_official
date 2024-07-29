# python BaseLine_finetune.py -c PC3 --n_epoch 50 --pretrain --finetune --balancesample

# python BaseLine_finetune.py -c PC3 -q AdversarialBIM --pretrain --finetune --balancesample

# python BaseLine_finetune.py -c PC3 -q LeastConfidence --pretrain --finetune
# python BaseLine_finetune.py -c PC3 -q MarginSampling --pretrain --finetune
# python BaseLine_finetune.py -c PC3 -q KMeansSampling --pretrain --finetune
# python BaseLine_finetune.py -c PC3 -q KCenterGreedy --pretrain --finetune
# python BaseLine_finetune.py -c PC3 -q BadgeSampling --pretrain --finetune
# python BaseLine_finetune.py -c PC3 -q BALDDropout --pretrain --finetune
# python BaseLine_finetune.py -c PC3 -q RandomSampling --pretrain --finetune
# python BaseLine_finetune.py -c PC3 -q LeastConfidence --pretrain --finetune --balancesample
# python BaseLine_finetune.py -c PC3 -q MarginSampling --pretrain --finetune --balancesample
# python BaseLine_finetune.py -c PC3 -q KMeansSampling --pretrain --finetune --balancesample
# python BaseLine_finetune.py -c PC3 -q KCenterGreedy --pretrain --finetune --balancesample
# python BaseLine_finetune.py -c PC3 -q BadgeSampling --pretrain --finetune --balancesample
# python BaseLine_finetune.py -c PC3 -q BALDDropout --pretrain --finetune --balancesample
# python BaseLine_finetune.py -c PC3 -q RandomSampling --pretrain --finetune --balancesample

python BaseLine_finetune.py -c PC3 -q AdversarialBIM-0.75-0.1-0.0005 --pretrain --finetune --balancesample
python BaseLine_finetune.py -c PC3 -q AdversarialBIM-0.8-0.1-0.0005 --pretrain --finetune --balancesample
python BaseLine_finetune.py -c PC3 -q AdversarialBIM-0.85-0.1-0.0005 --pretrain --finetune --balancesample
python BaseLine_finetune.py -c PC3 -q AdversarialBIM-0.9-0.1-0.0005 --pretrain --finetune --balancesample
python BaseLine_finetune.py -c PC3 -q AdversarialBIM-0.95-0.1-0.0005 --pretrain --finetune --balancesample