from train import Train
from model import RNN_custom

# mac
# dataroot = "/Users/akshita/Documents/Acads/data/sentiment_analysis/"
# path_to_saved_embeddings = "/Users/akshita/Documents/Acads/data/word2vec"

#vm
dataroot = "/mnt/cloudNAS4/akshita/Documents/datasets/sentiment_analysis/"
path_to_saved_embeddings = "/mnt/cloudNAS4/akshita/Documents/datasets/word2vec"

train = Train(dataroot, path_to_saved_embeddings=path_to_saved_embeddings)
hidden_size = 512
net = RNN_custom(300, hidden_size, 2, True, 2)
lr = 1e-5
momentum = 0.9
name = f"RNN_custom_{hidden_size}_{str(lr).replace('.', '_')}_{str(momentum).replace('.', '_')}"
train.train(net, name, data_size=400, lr=lr, momentum=momentum, epochs=1, batch_size=1)