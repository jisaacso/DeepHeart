from parser import PCG
from model import CNN
import sys

true_strs = {"True", "true", "t"}

def load_and_train_model(model_path, load_pretrained):
    pcg = PCG(model_path)

    if load_pretrained:
        pcg.load("/tmp")
    else:
        pcg.initialize_wav_data()

    cnn = CNN(pcg, epochs=100, dropout=0.5)
    cnn.train()

if __name__ == '__main__':
    data_path = sys.argv[1]

    load_pretrained = False
    if len(sys.argv) == 3:
        load_pretrained = sys.argv[2] in true_strs

    load_and_train_model(data_path, load_pretrained)