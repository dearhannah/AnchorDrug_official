# python ActiveLearning.py -a=RandomSampling -s=0 -q=30 -b=10 -d=LINCS --seed=4666 -t=1 -g=0
# python ActiveLearning.py -a=BadgeSampling -s=0 -q=30 -b=10 -d=LINCS --seed=4666 -t=1 -g=0
# python ActiveLearning.py -a=KCenterGreedy -s=0 -q=30 -b=10 -d=LINCS --seed=4666 -t=1 -g=0
# python ActiveLearning.py -a=MarginSampling -s=0 -q=30 -b=10 -d=LINCS --seed=4666 -t=1 -g=0
# python ActiveLearning.py -a=KMeansSamplingGPU -s=0 -q=30 -b=10 -d=LINCS --seed=4666 -t=1 -g=0
# python ActiveLearning.py -a=BALDDropout -s=0 -q=30 -b=10 -d=LINCS --seed=4666 -t=1 -g=0
python ActiveLearning.py -a=LeastConfidence -s=0 -q=30 -b=10 -d=LINCS -c MCF7 --seed=4666 -t=3 -g=0
python ActiveLearning.py -a=LeastConfidence -s=0 -q=30 -b=10 -d=LINCS -c A549 --seed=4666 -t=3 -g=0
python ActiveLearning.py -a=LeastConfidence -s=0 -q=30 -b=10 -d=LINCS -c PC3 --seed=4666 -t=3 -g=0