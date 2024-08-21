# AnchorDrug
This is the official directory of paper AnchorDrug.

## Conda Environment
conda create -n <environment-name> --file req.txt

## Code

### Active Learning
1. Original active learning code is in AnchorDrug/ActiveLearning_original. An Example command is:

`python ActiveLearning.py -a=LeastConfidence -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4996 -t=5 -g=6`

2. Active learning code for S1 is in AnchorDrug/ActiveLearning_S1. An Example command is:

`python ActiveLearning.py -a=LeastConfidence -q=100 -b=10 -g=7`

3. Active learning code for S2 is in AnchorDrug/ActiveLearning_S2. An Example command is:

`python ActiveLearning.py -a=LeastConfidence -q=100 -b=10 -g=7 -c MCF7`

### Finetune
`python BaseLine_finetune.py -c A549 -q LeastConfidence --pretrain --finetune --balancesample`
