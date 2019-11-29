import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

if __name__ == '__main__':
    train_data = pd.read_csv('train.csv', nrows=10)

    input = train_data.values.tolist()[9][1:]
    
    plt.figure(figsize=(28, 28))
    img = list()
    for y in range(28):
        img.append(input[y*28:y*28 + 28])

    plt.imshow(img)
    plt.savefig('sample.png')
