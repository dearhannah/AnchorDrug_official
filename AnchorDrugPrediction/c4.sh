# python anchor_drug_test.py --cell_line MCF7 --querymethod KMeans
# python anchor_drug_test.py --cell_line PC3 --querymethod KMeans
# python anchor_drug_test.py --cell_line A549 --querymethod KMeans

python anchor_drug_test.py --cell_line MCF7 --querymethod naiveUncertainty1
python anchor_drug_test.py --cell_line PC3 --querymethod naiveUncertainty1
python anchor_drug_test.py --cell_line A549 --querymethod naiveUncertainty1

python anchor_drug_test.py --cell_line MCF7 --querymethod naiveUncertainty2
python anchor_drug_test.py --cell_line PC3 --querymethod naiveUncertainty2
python anchor_drug_test.py --cell_line A549 --querymethod naiveUncertainty2