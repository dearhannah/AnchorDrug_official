# python ActiveLearning.py -a=LeastConfidence -b=5 -g=3
# python ActiveLearning.py -a=MarginSampling -b=5 -g=3
# python ActiveLearning.py -a=KMeansSampling -b=5 -g=3
# python ActiveLearning.py -a=KCenterGreedy -b=5 -g=3
# python ActiveLearning.py -a=BadgeSampling -b=5 -g=3
# python ActiveLearning.py -a=BALDDropout -b=5 -g=3
# python ActiveLearning.py -a=AdversarialBIM -b=5 -g=3
# python ActiveLearning.py -a=RandomSampling -b=5 -g=3

# python ActiveLearning.py -a=LeastConfidence -b=5 -q=100 -g=0
# python ActiveLearning.py -a=MarginSampling -b=5 -q=100 -g=1
# python ActiveLearning.py -a=KMeansSampling -b=5 -q=100 -g=2
# python ActiveLearning.py -a=KCenterGreedy -b=5 -q=100 -g=3
# python ActiveLearning.py -a=BadgeSampling -b=5 -q=100 -g=4
# python ActiveLearning.py -a=BALDDropout -b=5 -q=100 -g=5
# python ActiveLearning.py -a=AdversarialBIM -b=5 -q=100 -g=6
# python ActiveLearning.py -a=RandomSampling -b=5 -q=100 -g=7

# python ActiveLearning.py -a=RandomSampling -s=0 -q=810 -b=30 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=0
# python ActiveLearning.py -a=RandomSampling -s=0 -q=957 -b=30 -d=LINCS -c PC3 --seed=4996 -t=5 -g=1
# python ActiveLearning.py -a=RandomSampling -s=0 -q=642 -b=30 -d=LINCS -c A549 --seed=4786 -t=5 -g=2

# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=0 --bimratio 0.5
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4996 -t=5 -g=1 --bimratio 0.5
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c A549 --seed=4786 -t=5 -g=2 --bimratio 0.5

# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=0 --bimratio 0.6
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4996 -t=5 -g=1 --bimratio 0.6
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c A549 --seed=4786 -t=5 -g=2 --bimratio 0.6

# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=0 --bimratio 0.7
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4996 -t=5 -g=1 --bimratio 0.7
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c A549 --seed=4786 -t=5 -g=2 --bimratio 0.7

# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=0 --bimratio 0.8
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4996 -t=5 -g=1 --bimratio 0.8
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c A549 --seed=4786 -t=5 -g=2 --bimratio 0.8

# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=0 --bimratio 0.9
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4996 -t=5 -g=1 --bimratio 0.9
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c A549 --seed=4786 -t=5 -g=2 --bimratio 0.9

# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=5 --bimeps 7e-4
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=5 --bimeps 3e-4
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=5 --bimeps 5e-4
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=5 --bimeps 9e-4
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=5 --bimeps 11e-4
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=5 --bimeps 13e-4
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=5 --bimeps 15e-4

# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c A549 --seed=4678 -t=5 -g=7 --bimeps 7e-4
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c A549 --seed=4678 -t=5 -g=7 --bimeps 3e-4
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c A549 --seed=4678 -t=5 -g=7 --bimeps 5e-4
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c A549 --seed=4678 -t=5 -g=7 --bimeps 9e-4
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c A549 --seed=4678 -t=5 -g=5 --bimeps 11e-4
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c A549 --seed=4678 -t=5 -g=5 --bimeps 13e-4
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c A549 --seed=4678 -t=5 -g=5 --bimeps 15e-4

# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4678 -t=5 -g=6 --bimeps 7e-4
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4678 -t=5 -g=6 --bimeps 3e-4
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4678 -t=5 -g=6 --bimeps 5e-4
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4678 -t=5 -g=6 --bimeps 9e-4
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4678 -t=5 -g=5 --bimeps 11e-4
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4678 -t=5 -g=5 --bimeps 13e-4
python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4678 -t=5 -g=5 --bimeps 15e-4