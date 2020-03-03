import torch
import data, model, config

torch.backends.cudnn.benchmark = True

def main():
    path = 'logs/'
    files = []
    file = '2020-02-05_00:29:21.pth'
    results = torch.load(f=path+file, map_location='cpu')
    train_loader = data.get_loader(train=True)
    test_loader = data.get_loader(test=True)

    net = model.Net(train_loader.dataset.num_tokens).cuda()
    net.load_state_dict(results['weights'])
    print(net)

if __name__ == '__main__':
    main()