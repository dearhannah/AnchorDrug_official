# python ActiveLearning.py -a=LeastConfidence -b=5 -q=100 -g=4
# python ActiveLearning.py -a=MarginSampling -b=5 -q=100 -g=4
# python ActiveLearning.py -a=BadgeSampling -b=5 -q=100 -g=5
# python ActiveLearning.py -a=BALDDropout -b=5 -q=100 -g=5
python ActiveLearning.py -a=AdversarialBIM -b=5 -q=100 -g=7
python ActiveLearning.py -a=RandomSampling -b=5 -q=100 -g=7
python ActiveLearning.py -a=KMeansSampling -b=5 -q=100 -g=6
python ActiveLearning.py -a=KCenterGreedy -b=5 -q=100 -g=6