import urllib, requests
import zipfile
import os, sys, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

def delete_dir():
    shutil.rmtree('dataset/Images/train2014')

def download():
    url = 'http://images.cocodataset.org/zips/test2015.zip'
    urllib.request.urlretrieve(url, "dataset/Images/test2015.zip")
    # url = 'http://images.cocodataset.org/zips/val2014.zip'
    # urllib.request.urlretrieve(url, "dataset/Images/val2014.zip")
    # url = 'http://images.cocodataset.org/zips/train2014.zip'
    # urllib.request.urlretrieve(url, "dataset/Images/train2014.zip")

def unzip():
    # with zipfile.ZipFile('dataset/Images/train2014.zip', 'r') as zf:
    #     zf.extractall(path='dataset/Images')
    # with zipfile.ZipFile('dataset/Images/val2014.zip', 'r') as zf:
    #     zf.extractall(path='dataset/Images')
    # with zipfile.ZipFile('dataset/Images/test2015.zip', 'r') as zf:
    #     zf.extractall(path='dataset/Images')
    with zipfile.ZipFile('dataset/Annotations_Train_mscoco.zip', 'r') as zf, \
         zipfile.ZipFile('dataset/Questions_Train_mscoco.zip', 'r') as zf2:
        zf.extractall(path='dataset')
        zf2.extractall(path='dataset')

# download()
# unzip()
def tb_scaler():
    
    
    writer = SummaryWriter(log_dir='logs')
    for epoch in range(100):
        writer.add_scalar('scalar/test', np.random.rand(), epoch)
        writer.add_scalars('scalar/scalars_test', {'sinx': np.sin(epoch), 'cosx': np.cos(epoch)}, epoch)
    
    writer.close()


def tb_graph1():
    class Net1(nn.Module):
        def __init__(self):
            super(Net1, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)
            self.bn = nn.BatchNorm2d(20)

        def forward(self, x):
            x = F.max_pool2d(self.conv1(x), 2)
            x = F.relu(x) + F.relu(-x)
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = self.bn(x)
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            x = F.softmax(x, dim=1)
            return x

    dummy_input = torch.rand(13, 1, 28, 28)

    model = Net1()
    with SummaryWriter(log_dir='logs', comment='Net1') as w:
        w.add_graph(model, (dummy_input,))

def tb_graph2():
    import tensorflow as tf

    sess = tf.InteractiveSession()
    a = tf.Variable(0, name="a")
    b = tf.Variable(1, name="b")
    c = tf.add(a, b)

    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("logs", sess.graph)

def tb_train():

    input_size = 1
    output_size = 1
    num_epoches = 60
    learning_rate = 0.01
    writer = SummaryWriter(log_dir='logs', comment='Linear')
    x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                        [9.779], [6.182], [7.59], [2.167], [7.042],
                        [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
    y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                        [3.366], [2.596], [2.53], [1.221], [2.827],
                        [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

    model = nn.Linear(input_size, output_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epoches):
        inputs = torch.from_numpy(x_train)
        targets = torch.from_numpy(y_train)

        output = model(inputs)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 保存loss的数据与epoch数值
        writer.add_scalar('Train', loss, epoch)
        if (epoch + 1) % 5 == 0:
            print('Epoch {}/{},loss:{:.4f}'.format(epoch + 1, num_epoches, loss.item()))

    # 将model保存为graph
    writer.add_graph(model, (inputs,))

    predicted = model(torch.from_numpy(x_train)).detach().numpy()
    plt.plot(x_train, y_train, 'ro', label='Original data')
    plt.plot(x_train, predicted, label='Fitted line')
    plt.legend()
    plt.show()
    writer.close()

def tb_train2():
    import torchvision.utils as vutils
    import torchvision.models as models
    from torchvision import datasets

    resnet18 = models.resnet18(False)
    writer = SummaryWriter()
    sample_rate = 44100
    freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]

    for n_iter in range(100):
        dummy_s1 = torch.rand(1)
        dummy_s2 = torch.rand(1)
        # data grouping by `slash`
        writer.add_scalar('data/scalar1', dummy_s1[0], n_iter)
        writer.add_scalar('data/scalar2', dummy_s2[0], n_iter)

        writer.add_scalars('data/scalar_group', {'xsinx': n_iter * np.sin(n_iter),
                                                 'xcosx': n_iter * np.cos(n_iter),
                                                 'arctanx': np.arctan(n_iter)}, n_iter)

        dummy_img = torch.rand(32, 3, 64, 64)  # output from network
        if n_iter % 10 == 0:
            x = vutils.make_grid(dummy_img, normalize=True, scale_each=True)
            writer.add_image('Image', x, n_iter)

            dummy_audio = torch.zeros(sample_rate * 2)
            for i in range(x.size(0)):
                # amplitude of sound should in [-1, 1]
                dummy_audio[i] = np.cos(freqs[n_iter // 10] * np.pi * float(i) / float(sample_rate))
            writer.add_audio('myAudio', dummy_audio, n_iter, sample_rate=sample_rate)

            writer.add_text('Text', 'text logged at step:' + str(n_iter), n_iter)

            for name, param in resnet18.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)

            # needs tensorboard 0.4RC or later
            writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(100), n_iter)

    dataset = datasets.MNIST('mnist', train=False, download=True)
    images = dataset.test_data[:100].float()
    label = dataset.test_labels[:100]

    features = images.view(100, 784)
    writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
# tb_scaler()
tb_graph1()
# tb_train()