# AnchorDrug
## Active Learning
### with imblanced pretrain model, active learning with imblanced data
batch32_epoch20_imbalance/
imbalance_batch32_epoch20_all_data/
### with imblanced pretrain model, active learning with blanced data
balance_batch32_epoch20_all_data/
### advbim ratio tuning, with imblanced pretrain model, active learning with imblanced data
advbim_ratio/
### advbim distance tuning, with imblanced pretrain model, active learning with imblanced data
advbim_ratio/
## Finetune
### with imblanced pretrain model, active learning with imblanced data


mkdir balance_batch32_epoch20_all_data
mv *RandomSa* balance_batch32_epoch20_all_data

mkdir advbim_ratio
mv *_AdversarialBIM* advbim_ratio

mkdir advbim_dist && mv *_AdversarialBIM* advbim_dist
