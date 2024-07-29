# python BaseLine_finetune.py -c A549 --n_epoch 50 --pretrain --finetune --balancesample

# python BaseLine_finetune.py -c A549 -q AdversarialBIM --pretrain --finetune --balancesample

# python BaseLine_finetune.py -c A549 -q LeastConfidence --pretrain --finetune
# python BaseLine_finetune.py -c A549 -q MarginSampling --pretrain --finetune
# python BaseLine_finetune.py -c A549 -q KMeansSampling --pretrain --finetune
# python BaseLine_finetune.py -c A549 -q KCenterGreedy --pretrain --finetune
# python BaseLine_finetune.py -c A549 -q BadgeSampling --pretrain --finetune
# python BaseLine_finetune.py -c A549 -q BALDDropout --pretrain --finetune
# python BaseLine_finetune.py -c A549 -q RandomSampling --pretrain --finetune
# python BaseLine_finetune.py -c A549 -q LeastConfidence --pretrain --finetune --balancesample
# python BaseLine_finetune.py -c A549 -q MarginSampling --pretrain --finetune --balancesample
# python BaseLine_finetune.py -c A549 -q KMeansSampling --pretrain --finetune --balancesample
# python BaseLine_finetune.py -c A549 -q KCenterGreedy --pretrain --finetune --balancesample
# python BaseLine_finetune.py -c A549 -q BadgeSampling --pretrain --finetune --balancesample
# python BaseLine_finetune.py -c A549 -q BALDDropout --pretrain --finetune --balancesample
# python BaseLine_finetune.py -c A549 -q RandomSampling --pretrain --finetune --balancesample

python BaseLine_finetune.py -c A549 -q AdversarialBIM-0.75-0.1-0.0005 --pretrain --finetune --balancesample
python BaseLine_finetune.py -c A549 -q AdversarialBIM-0.8-0.1-0.0005 --pretrain --finetune --balancesample
python BaseLine_finetune.py -c A549 -q AdversarialBIM-0.85-0.1-0.0005 --pretrain --finetune --balancesample
python BaseLine_finetune.py -c A549 -q AdversarialBIM-0.9-0.1-0.0005 --pretrain --finetune --balancesample
python BaseLine_finetune.py -c A549 -q AdversarialBIM-0.95-0.1-0.0005 --pretrain --finetune --balancesample