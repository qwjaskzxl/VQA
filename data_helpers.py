import urllib, requests
import zipfile

def download():
    url = 'http://images.cocodataset.org/zips/test2015.zip'
    urllib.request.urlretrieve(url, "dataset/Images/test2015.zip")
    # url = 'http://images.cocodataset.org/zips/val2014.zip'
    # urllib.request.urlretrieve(url, "dataset/Images/val2014.zip")
    # url = 'http://images.cocodataset.org/zips/train2014.zip'
    # urllib.request.urlretrieve(url, "dataset/Images/train2014.zip")

def unzip():
    with zipfile.ZipFile('dataset/Images/train2014.zip', 'r') as zf:
        zf.extractall(path='dataset/Images')
    with zipfile.ZipFile('dataset/Images/val2014.zip', 'r') as zf:
        zf.extractall(path='dataset/Images')
    with zipfile.ZipFile('dataset/Images/test2015.zip', 'r') as zf:
        zf.extractall(path='dataset/Images')

# download()
# unzip()