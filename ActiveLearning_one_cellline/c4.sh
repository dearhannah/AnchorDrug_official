# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=3 --bimdis 0.9
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4996 -t=5 -g=3 --bimdis 0.9
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c A549 --seed=4786 -t=5 -g=3 --bimdis 0.9


python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=3 -g=5 --bimeps 0.0005 --bimratio 0.9
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=3 -g=5 --bimeps 0.0005 --bimratio 0.95

python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4678 -t=3 -g=5 --bimeps 0.0005 --bimratio 0.9
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4678 -t=3 -g=5 --bimeps 0.0005 --bimratio 0.95

python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c A549 --seed=4678 -t=3 -g=5 --bimeps 0.0005 --bimratio 0.9
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c A549 --seed=4678 -t=3 -g=5 --bimeps 0.0005 --bimratio 0.95