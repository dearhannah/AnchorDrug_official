python pretrain_base_model_combine_GO_terms_finetune_anchor_drugs_baseline_GPS_predictable_genes_HQ_LINCS_with_internal_val_han_1.py --n_drug 30 --query LeastConfidence
python pretrain_base_model_combine_GO_terms_finetune_anchor_drugs_baseline_GPS_predictable_genes_HQ_LINCS_with_internal_val_han_1.py --n_drug 30 --query MarginSampling
python pretrain_base_model_combine_GO_terms_finetune_anchor_drugs_baseline_GPS_predictable_genes_HQ_LINCS_with_internal_val_han_1.py --n_drug 30 --query KMeansSampling
python pretrain_base_model_combine_GO_terms_finetune_anchor_drugs_baseline_GPS_predictable_genes_HQ_LINCS_with_internal_val_han_1.py --n_drug 30 --query KCenterGreedy
python pretrain_base_model_combine_GO_terms_finetune_anchor_drugs_baseline_GPS_predictable_genes_HQ_LINCS_with_internal_val_han_1.py --n_drug 30 --query BadgeSampling
python pretrain_base_model_combine_GO_terms_finetune_anchor_drugs_baseline_GPS_predictable_genes_HQ_LINCS_with_internal_val_han_1.py --n_drug 30 --query BALDDropout
python pretrain_base_model_combine_GO_terms_finetune_anchor_drugs_baseline_GPS_predictable_genes_HQ_LINCS_with_internal_val_han_1.py --n_drug 30 --query AdversarialBIM
# python pretrain_base_model_combine_GO_terms_finetune_anchor_drugs_baseline_GPS_predictable_genes_HQ_LINCS_with_internal_val_han_0.py --n_drug 30 --query LeastConfidence


# python ActiveLearning.py -a=LeastConfidence -b=5 -g=3
# python ActiveLearning.py -a=MarginSampling -b=5 -g=3
# python ActiveLearning.py -a=KMeansSampling -b=5 -g=3
# python ActiveLearning.py -a=KCenterGreedy -b=5 -g=3
# python ActiveLearning.py -a=BadgeSampling -b=5 -g=3
# python ActiveLearning.py -a=BALDDropout -b=5 -g=3
# python ActiveLearning.py -a=AdversarialBIM -b=5 -g=3
# python ActiveLearning.py -a=RandomSampling -b=5 -g=3