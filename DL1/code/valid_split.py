from sklearn.model_selection import train_test_split
import glob
from pathlib import Path

paths = []
for filename in glob.iglob('./dataset/'+'**/*.jpg', recursive=True):
    paths.append(Path(filename))

y = paths
x_train, x_test, y_train, y_test = train_test_split(paths, y , test_size=0.1, random_state=0)

train_names = []
for i in x_train:
    f = list(i.parts)
    f.insert(2,'train')
    i.rename('/'.join(f))
test_names = []
for i in x_test:
    f = list(i.parts)
    f.insert(2,'validation')
    i.rename('/'.join(f))
