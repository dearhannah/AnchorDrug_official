# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=0 --bimratio 0.5
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=0 --bimratio 0.6
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=0 --bimratio 0.7
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=0 --bimratio 0.8
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=0 --bimratio 0.9

# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=0 --bimdis 0.5
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=0 --bimdis 0.6
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=0 --bimdis 0.7
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=0 --bimdis 0.8

# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=5 --bimeps 2e-4
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=5 --bimeps 3e-4
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=5 --bimeps 4e-4
# python ActiveLearning.py -a=AdversarialBIM -s=0 -q=100 -b=10 -d=LINCS -c MCF7 --seed=4678 -t=5 -g=5 --bimeps 5e-4

python ActiveLearning.py -a=LeastConfidence -q=100 -b=10 -g=7 -c MCF7
python ActiveLearning.py -a=MarginSampling -q=100 -b=10 -g=7 -c MCF7
python ActiveLearning.py -a=KMeansSampling -q=100 -b=10 -g=7 -c MCF7
python ActiveLearning.py -a=KCenterGreedy -q=100 -b=10 -g=7 -c MCF7
python ActiveLearning.py -a=BadgeSampling -q=100 -b=10 -g=7 -c MCF7
python ActiveLearning.py -a=BALDDropout -q=100 -b=10 -g=7 -c MCF7
python ActiveLearning.py -a=RandomSampling -q=100 -b=10 -g=7 -c MCF7