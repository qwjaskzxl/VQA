import sys
import torch
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt


def main():
    # path = sys.argv[1]
    path = 'logs/'
    file = '2020-02-01_23:39:49.pth'
    results = torch.load(path+file)

    val_acc = torch.FloatTensor(results['tracker']['val_acc'])
    val_acc = val_acc.mean(dim=1).numpy()

    plt.figure()
    plt.plot(val_acc)
    plt.show()
    plt.savefig('val_acc.png')

if __name__ == '__main__':
    main()