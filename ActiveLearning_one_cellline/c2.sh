# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4996 -t=5 -g=1 --bimratio 0.5
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4996 -t=5 -g=1 --bimratio 0.6
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4996 -t=5 -g=1 --bimratio 0.7
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4996 -t=5 -g=1 --bimratio 0.8
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4996 -t=5 -g=1 --bimratio 0.9

# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4996 -t=5 -g=1 --bimdis 0.5
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4996 -t=5 -g=1 --bimdis 0.6
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4996 -t=5 -g=1 --bimdis 0.7
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4996 -t=5 -g=1 --bimdis 0.8

# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4678 -t=5 -g=6 --bimeps 2e-4
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4678 -t=5 -g=6--bimeps 3e-4
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4678 -t=5 -g=6 --bimeps 4e-4
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c PC3 --seed=4678 -t=5 -g=6 --bimeps 5e-4

python ActiveLearning.py -a=LeastConfidence -q=100 -b=10 -g=7 -c PC3
python ActiveLearning.py -a=MarginSampling -q=100 -b=10 -g=7 -c PC3
python ActiveLearning.py -a=KMeansSampling -q=100 -b=10 -g=7 -c PC3
python ActiveLearning.py -a=KCenterGreedy -q=100 -b=10 -g=7 -c PC3
python ActiveLearning.py -a=BadgeSampling -q=100 -b=10 -g=7 -c PC3
python ActiveLearning.py -a=BALDDropout -q=100 -b=10 -g=7 -c PC3
python ActiveLearning.py -a=RandomSampling -q=100 -b=10 -g=7 -c PC3