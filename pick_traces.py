import glob
import random

random.seed(15213)

for f in random.sample(glob.glob("Traces/*"), 50):
    print(f)