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
>>> ActiveLearning_one_cellline/logfile/new_advbim_ratio
### finetune to compare new advbim code, hyperparameter ratio
```
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
>>> /egr/research-aidd/menghan1/AnchorDrug/ActiveLearning_one_cellline/druglist/new_advbim_ratio/