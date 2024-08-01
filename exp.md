## adv bim hyper parameter

### I updated the advbim code for ActiveLearning_one_cellline: 1 removed dis parameter. 2 add gradient clip. 
### then I did al experiments to test bimratio from 0.5 - 0.9
```
@Jul27 
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=3 -g=5 --bimratio 0.5
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=3 -g=5 --bimratio 0.6
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=3 -g=5 --bimratio 0.7
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=3 -g=5 --bimratio 0.8
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=3 -g=5 --bimratio 0.9
```
> ActiveLearning_one_cellline/logfile/new_advbim_ratio

### finetune to compare new advbim code, hyperparameter ratio
```
@jul28
python BaseLine_finetune.py -c MCF7 -q AdversarialBIM-0.5-0.1-0.001 --pretrain --finetune --balancesample
wandb link --> https://wandb.ai/menghan/Anchor%20Drug%20Project/runs/xj2dcejj
python BaseLine_finetune.py -c MCF7 -q AdversarialBIM-0.6-0.1-0.001 --pretrain --finetune --balancesample
wandb link --> https://wandb.ai/menghan/Anchor%20Drug%20Project/runs/aoqd4oon
python BaseLine_finetune.py -c MCF7 -q AdversarialBIM-0.7-0.1-0.001 --pretrain --finetune --balancesample
wandb link --> https://wandb.ai/menghan/Anchor%20Drug%20Project/runs/tl21er4a
python BaseLine_finetune.py -c MCF7 -q AdversarialBIM-0.8-0.1-0.001 --pretrain --finetune --balancesample
wandb link --> https://wandb.ai/menghan/Anchor%20Drug%20Project/runs/08wupt37
python BaseLine_finetune.py -c MCF7 -q AdversarialBIM-0.9-0.1-0.001 --pretrain --finetune --balancesample
wandb link --> https://wandb.ai/menghan/Anchor%20Drug%20Project/runs/z29gkp6k
```
> /egr/research-aidd/menghan1/AnchorDrug/resultBaseLine/new_advBIM_ratio

### I did al experiments to test bimratio from 0.75, 0.8, 0.85, 0.9, 0.95
```
@jul29
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=3 -g=5 --bimeps 0.0005 --bimratio 0.75
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=3 -g=5 --bimeps 0.0005 --bimratio 0.8
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=3 -g=5 --bimeps 0.0005 --bimratio 0.85
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=3 -g=5 --bimeps 0.0005 --bimratio 0.9
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=3 -g=5 --bimeps 0.0005 --bimratio 0.95
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4678 -t=3 -g=5 --bimeps 0.0005 --bimratio 0.75
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4678 -t=3 -g=5 --bimeps 0.0005 --bimratio 0.8
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4678 -t=3 -g=5 --bimeps 0.0005 --bimratio 0.85
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4678 -t=3 -g=5 --bimeps 0.0005 --bimratio 0.9
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4678 -t=3 -g=5 --bimeps 0.0005 --bimratio 0.95
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c A549 --seed=4678 -t=3 -g=5 --bimeps 0.0005 --bimratio 0.75
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c A549 --seed=4678 -t=3 -g=5 --bimeps 0.0005 --bimratio 0.8
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c A549 --seed=4678 -t=3 -g=5 --bimeps 0.0005 --bimratio 0.85
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c A549 --seed=4678 -t=3 -g=5 --bimeps 0.0005 --bimratio 0.9
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c A549 --seed=4678 -t=3 -g=5 --bimeps 0.0005 --bimratio 0.95
```
> ActiveLearning_one_cellline/logfile/new_advbim_ratio

### finetune to compare new advbim code, hyperparameter ratio
```
@jul29
python BaseLine_finetune.py -c MCF7 -q AdversarialBIM-0.75-0.1-0.0005 --pretrain --finetune --balancesample
wandb link --> 
python BaseLine_finetune.py -c MCF7 -q AdversarialBIM-0.8-0.1-0.0005 --pretrain --finetune --balancesample
wandb link --> 
python BaseLine_finetune.py -c MCF7 -q AdversarialBIM-0.85-0.1-0.0005 --pretrain --finetune --balancesample
wandb link --> 
python BaseLine_finetune.py -c MCF7 -q AdversarialBIM-0.9-0.1-0.0005 --pretrain --finetune --balancesample
wandb link --> 
python BaseLine_finetune.py -c MCF7 -q AdversarialBIM-0.95-0.1-0.0005 --pretrain --finetune --balancesample
wandb link --> 

python BaseLine_finetune.py -c PC3 -q AdversarialBIM-0.75-0.1-0.0005 --pretrain --finetune --balancesample
wandb link --> 
python BaseLine_finetune.py -c PC3 -q AdversarialBIM-0.8-0.1-0.0005 --pretrain --finetune --balancesample
wandb link --> 
python BaseLine_finetune.py -c PC3 -q AdversarialBIM-0.85-0.1-0.0005 --pretrain --finetune --balancesample
wandb link --> 
python BaseLine_finetune.py -c PC3 -q AdversarialBIM-0.9-0.1-0.0005 --pretrain --finetune --balancesample
wandb link --> 
python BaseLine_finetune.py -c PC3 -q AdversarialBIM-0.95-0.1-0.0005 --pretrain --finetune --balancesample
wandb link --> 

python BaseLine_finetune.py -c A549 -q AdversarialBIM-0.75-0.1-0.0005 --pretrain --finetune --balancesample
wandb link --> 
python BaseLine_finetune.py -c A549 -q AdversarialBIM-0.8-0.1-0.0005 --pretrain --finetune --balancesample
wandb link --> 
python BaseLine_finetune.py -c A549 -q AdversarialBIM-0.85-0.1-0.0005 --pretrain --finetune --balancesample
wandb link --> 
python BaseLine_finetune.py -c A549 -q AdversarialBIM-0.9-0.1-0.0005 --pretrain --finetune --balancesample
wandb link --> 
python BaseLine_finetune.py -c A549 -q AdversarialBIM-0.95-0.1-0.0005 --pretrain --finetune --balancesample
wandb link --> 
```
> /egr/research-aidd/menghan1/AnchorDrug/resultBaseLine/new_advBIM_ratio


## active learning converge curve
```
python ActiveLearning.py -a=RandomSampling -s=0 -q=810 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=3 -bs
python ActiveLearning.py -a=RandomSampling -s=0 -q=957 -b=10 -d=LINCS -c PC3 --seed=4996 -t=3 -bs
python ActiveLearning.py -a=RandomSampling -s=0 -q=642 -b=10 -d=LINCS -c A549 --seed=4786 -t=3 -bs
```


## baseline (MOA, Cluster, Random selction) finetune 30/100
```
python BaseLine_finetune.py -c A549 --pretrain --finetune --balancesample
python BaseLine_finetune.py -c PC3 --pretrain --finetune --balancesample
python BaseLine_finetune.py -c MCF7 --pretrain --finetune --balancesample

```
> /egr/research-aidd/menghan1/AnchorDrug/resultBaseLine/baselines

## model size v.s. finetune performance
