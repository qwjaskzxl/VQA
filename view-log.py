import sys
import torch
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image

def main():
    # path = sys.argv[1] #cmd：python view-log.py path
    path = 'logs/'
    files = ['best3 2020-02-07_18:54:45.pth','2020-02-11_23:40:08.pth', '2020-02-12_00:36:36.pth',]
    plt.figure()

    for file in files:
        results = torch.load(f=path+file, map_location='cpu')

        # train_acc = torch.FloatTensor(results['tracker']['train_acc'])
        # train_acc = train_acc.mean(dim=1).numpy()
        # plt.plot(train_acc)
        print(results['config'])
        val_acc = torch.FloatTensor(results['tracker']['val_acc'])
        val_acc = val_acc.mean(dim=1).numpy()
        plt.plot(val_acc)
    # plt.show()
    # plt.savefig(path+'%s.png'%file)
    plt.savefig(path+'1.png')

if __name__ == '__main__':
    main()
    # img = Image.open('logs/1.png')
    # print(img)
    # img.show()