import sys
import torch
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image

def main():
    # path = sys.argv[1]
    path = 'logs/'
    file = '2020-02-04_11:37:20.pth'
    results = torch.load(f=path+file, map_location='cpu')

    train_acc = torch.FloatTensor(results['tracker']['train_acc'])
    train_acc = train_acc.mean(dim=1).numpy()

    val_acc = torch.FloatTensor(results['tracker']['val_acc'])
    val_acc = val_acc.mean(dim=1).numpy()

    plt.figure()
    plt.plot(train_acc)
    plt.plot(val_acc)
    # plt.show()
    # plt.savefig(path+'%s.png'%file)
    plt.savefig('logs/1.png')

if __name__ == '__main__':
    main()
    img = Image.open('logs/1.png')
    print(img)
    img.show()