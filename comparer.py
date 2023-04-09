import numpy as np
import matplotlib.pyplot as plt
import re

filename = 'heat0.csv'
file1optimization = 'heat.csv'

# Read the file


def read_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            data.append(line.split())
    return data


def main():
    # load original data
    data = read_file(filename)
    data = np.array(data)
    # load optimized data
    filesnames = ["heat" + str(i) + ".csv" for i in range(1, 7)]
    TEST = []
    Xs = []
    Ys = []
    for file in filesnames:
        d = read_file(file)
        d = np.array(d)
        x0 = [x[0] for x in d]
        y0 = [float(x[1]) for x in d]
        Xs.append(x0)
        Ys.append(y0)

    X = [x[0] for x in data]
    Y = [float(x[1]) for x in data]

    plt.plot(X, Y, label='Original', color='red')
    for i in range(len(Xs)):
        plt.plot(Xs[i], Ys[i], label='Optimized' + str(i))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
