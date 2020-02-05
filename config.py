# paths
qa_path = './dataset'  # directory containing the question and annotation jsons
train_path = 'dataset/Images/train2014'  # directory of training images
val_path = 'dataset/Images/val2014'  # directory of validation images
test_path = 'dataset/Images/test2015'  # directory of test images
preprocessed_path = './dataset/resnet152_pre_byori-14x14.h5'  # path where preprocessed features are saved to and loaded from
vocabulary_path = './dataset/vocab.json'  # path where the used vocabularies for question and answers are saved to
glove_path = './dataset/glove.6B.300d.json'
task = 'OpenEnded'
dataset = 'mscoco'

# preprocess config
preprocess_batch_size = 64 # author：64
image_size = 448  # scale shorter end of image to this size and centre crop
output_size = image_size // 32  # size of the feature maps after processing through a network
output_features = 2048  # number of feature maps thereof
central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping

# model config
embedding_features = 300
question_features = 2048
attn_mid_features = 2048
Classifier_mid_features = 1024

# training config
epochs = 21
batch_size = 128
initial_lr = 1e-3  # default Adam lr
lr_halflife = 50000  # in iterations
data_workers = 40
max_answers = 3000

cuda_device = '3'
device_ids = [0,3]