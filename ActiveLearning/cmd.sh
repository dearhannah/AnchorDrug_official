python ActiveLearning.py -a=LeastConfidence -b=5 -g=3
python ActiveLearning.py -a=MarginSampling -b=5 -g=3
python ActiveLearning.py -a=KMeansSampling -b=5 -g=3
python ActiveLearning.py -a=KCenterGreedy -b=5 -g=3
python ActiveLearning.py -a=BadgeSampling -b=5 -g=3
python ActiveLearning.py -a=BALDDropout -b=5 -g=3
python ActiveLearning.py -a=AdversarialBIM -b=5 -g=3
python ActiveLearning.py -a=RandomSampling -b=5 -g=3

python ActiveLearning.py -a=LeastConfidence -b=5 -q=100 -g=0
python ActiveLearning.py -a=MarginSampling -b=5 -q=100 -g=1
python ActiveLearning.py -a=KMeansSampling -b=5 -q=100 -g=2
python ActiveLearning.py -a=KCenterGreedy -b=5 -q=100 -g=3
python ActiveLearning.py -a=BadgeSampling -b=5 -q=100 -g=4
python ActiveLearning.py -a=BALDDropout -b=5 -q=100 -g=5
python ActiveLearning.py -a=AdversarialBIM -b=5 -q=100 -g=6
python ActiveLearning.py -a=RandomSampling -b=5 -q=100 -g=7