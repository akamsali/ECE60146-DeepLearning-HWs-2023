from DLStudio import DLStudio
from model import RNN_custom, UNI_GRU, BI_GRU

from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pickle
from tqdm import tqdm

dataroot = "/mnt/cloudNAS4/akshita/Documents/datasets/sentiment_analysis/"
path_to_saved_embeddings = "/mnt/cloudNAS4/akshita/Documents/datasets/word2vec"
dataset_archive_train = "sentiment_dataset_train_400.tar.gz"
dataset_archive_test = "sentiment_dataset_test_400.tar.gz"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dls = DLStudio(
                  dataroot = dataroot,
                  path_saved_model = "./saved_model",
                  momentum = 0.9,
                  learning_rate =  1e-5,
                  epochs = 1,
                  batch_size = 1,
                  classes = ('negative','positive'),
                  use_gpu = True if torch.cuda.is_available() else False,
              )

text_cl = DLStudio.TextClassificationWithEmbeddings( dl_studio = dls )

dataserver_train = DLStudio.TextClassificationWithEmbeddings.SentimentAnalysisDataset(
                                 train_or_test = 'train',
                                 dl_studio = dls,
                                 dataset_file = dataset_archive_train,
                                 path_to_saved_embeddings = path_to_saved_embeddings,
                   )
dataserver_test = DLStudio.TextClassificationWithEmbeddings.SentimentAnalysisDataset(
                                 train_or_test = 'test',
                                 dl_studio = dls,
                                 dataset_file = dataset_archive_test,
                                 path_to_saved_embeddings = path_to_saved_embeddings,
                  )
text_cl.dataserver_train = dataserver_train
text_cl.dataserver_test = dataserver_test

text_cl.load_SentimentAnalysisDataset(dataserver_train, dataserver_test)


def validation(net, name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.eval()
    true = []
    pred = []
    with torch.no_grad():
        for data in tqdm(text_cl.test_dataloader):
            # get a sample from the test loader
            review_tensor, sentiment = data["review"], data["sentiment"]

            # send review and sentiment tensor to cuda
            review_tensor = review_tensor.to(device)
            sentiment = sentiment.to(device)

            output = net(review_tensor)
            pred.append(torch.argmax(output).item())
            true.append(torch.argmax(sentiment).item())
        pickle.dump(true, open(f"./solutions/{name}_true.pkl", "wb"))
        pickle.dump(pred, open(f"./solutions/{name}_pred.pkl", "wb"))

    # return true, pred


def validate_and_conf_matrix(net, val_dataset, categories, name="Net") -> None:
    t, p = validation(net, val_dataset)
    cm = confusion_matrix(t, p)
    plt.figure()
    sns.heatmap(cm, annot=cm, xticklabels=categories, yticklabels=categories, fmt="g")
    plt.title(f"Confusion matrix for {name}, accuracy={accuracy_score(t, p)}")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.savefig(f"./solutions/cm_{name}.png")


net_custom = RNN_custom(300, 512, 1, True, 2)
net_custom.load_state_dict(
    torch.load(
        "./solutions/my_gru_final_model_4.pt",
        map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
)
validation(net_custom, "my_gru")

net_uni = UNI_GRU(300, 512, 2, True, 1)
net_uni.load_state_dict(
    torch.load(
        "./solutions/uni_model_4.pt",
        map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
)
validation(net_uni, "uni_gru")

net_bi = BI_GRU(300, 512, 2, True, 1)
net_bi.load_state_dict(
    torch.load(
        "./solutions/bi_model_0.pt",
        map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ) 
)
validation(net_bi, "bi_gru")


def plot_training_loss(training_loss_tally, name):
    plt.figure()
    plt.plot(training_loss_tally)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.savefig(f'./solutions/{name}_training_loss.png')

for m in ['uni', 'bi', 'my_gru']:
    with open(f'./solutions/{m}_training_log.csv', 'r') as csvFile:
        reader = csv.reader(csvFile)
        training_loss_tally = [float(row[2]) for row in reader]
    plot_training_loss(training_loss_tally, m)

