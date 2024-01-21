# python ActiveLearning.py -a=RandomSampling -s=0 -q=30 -b=10 -d=LINCS --seed=4666 -t=1 -g=0
# python ActiveLearning.py -a=BadgeSampling -s=0 -q=30 -b=10 -d=LINCS --seed=4666 -t=1 -g=0
# python ActiveLearning.py -a=KCenterGreedy -s=0 -q=30 -b=10 -d=LINCS --seed=4666 -t=1 -g=0
# python ActiveLearning.py -a=MarginSampling -s=0 -q=30 -b=10 -d=LINCS --seed=4666 -t=1 -g=0
# python ActiveLearning.py -a=KMeansSamplingGPU -s=0 -q=30 -b=10 -d=LINCS --seed=4666 -t=1 -g=0
# python ActiveLearning.py -a=BALDDropout -s=0 -q=30 -b=10 -d=LINCS --seed=4666 -t=1 -g=0

python ActiveLearning.py -a=KMeansSampling -b=10 -c MCF7 -g=1
python ActiveLearning.py -a=KMeansSampling -b=10 -c A549 -g=2
python ActiveLearning.py -a=KMeansSampling -b=10 -c PC3 -g=3

python ActiveLearning.py -a=KMeansSampling -b=5 -c MCF7 -g=1
python ActiveLearning.py -a=KMeansSampling -b=5 -c A549 -g=2
python ActiveLearning.py -a=KMeansSampling -b=5 -c PC3 -g=3

python ActiveLearning.py -a=LeastConfidence -b=10 -c MCF7 -g=1
python ActiveLearning.py -a=LeastConfidence -b=10 -c A549 -g=2
python ActiveLearning.py -a=LeastConfidence -b=10 -c PC3 -g=3

python ActiveLearning.py -a=LeastConfidence -b=5 -c MCF7 -g=1
python ActiveLearning.py -a=LeastConfidence -b=5 -c A549 -g=2
python ActiveLearning.py -a=LeastConfidence -b=5 -c PC3 -g=3