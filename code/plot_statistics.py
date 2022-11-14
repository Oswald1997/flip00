import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("../data/input/train.csv")


def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf

dataset['length'] = dataset['text'].str.len()
length = dataset['length']
mean = length.mean()
std = length.std()

x = np.arange(0, 200, 0.1)
y = normfun(x, mean, std)
plt.plot(x, y)

# 绘制数据集的直方图
plt.hist(length, bins=15, rwidth=0.9, density=True)
# plt.title('Length distribution')
plt.xlabel('Character Length')
plt.ylabel('Probability')

plt.show()



f = open('../data/output/token.data', 'rb')
tokenlist = pickle.load(f)

f.close()
tokennum = []
for i in range(7613):
    n = 0
    for element in tokenlist[i]:
        if (element != 0):
            n += 1
    tokennum.append(n)
# print(tokennum)
# temp = tokennum
# temp.sort()
# print(temp)
dataset['tokenlength'] = tokennum

length = dataset['tokenlength']
mean = length.mean()
std = length.std()

x = np.arange(0, 90, 0.1)
y = normfun(x, mean, std)
plt.plot(x, y)

# 绘制数据集的直方图
plt.hist(length, bins=15, rwidth=0.9, density=True)
# plt.title('Length distribution')
plt.xlabel('Token Length')
plt.ylabel('Probability')

plt.show()






