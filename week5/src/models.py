import torch.nn as nn
from torchvision import models
import fasttext
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class EmbeddingNetTextBERT(nn.Module):
    def __init__(self, model_name, out_features):
        super(EmbeddingNetTextBERT, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.out_features = out_features

    def forward(self, x):
        input_ids = self.tokenizer.batch_encode_plus(
            x,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )['input_ids'].to(self.model.device)
        outputs = self.model(input_ids)[0]
        outputs = outputs[:, 0, :]  # use the [CLS] token representation
        outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)  # normalize embeddings
        return outputs

class EmbeddingNetImage(nn.Module):
    def __init__(self, out_features, pretrained=True):  # dim_out_fc = 'as_image' or 'as_text'
        super(EmbeddingNetImage, self).__init__()

        self.model = models.resnet18(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, out_features=out_features)

    def forward(self, x):
        output = self.model(x)

        return output


class EmbeddingNetText(nn.Module):
    def __init__(self, weights, out_features):  # type = 'FastText' or 'BERT'
        super(EmbeddingNetText, self).__init__()
        self.model = fasttext.load_model(weights)

    def forward(self, x):
        outputs = np.empty((len(x), 300))
        # x is a tuple of N strings, need to return N feature vectors
        for i, text in enumerate(x):
            words_features = []
            words = text.split(" ")
            for word in words:
                output = self.model[word]
                words_features.append(output)
            outputs[i] = np.array(words_features).mean(axis=0)

        return outputs


class TripletNetIm2Text(nn.Module):
    def __init__(self, embedding_net_image, embedding_net_text):
        super(TripletNetIm2Text, self).__init__()
        self.embedding_net_image = embedding_net_image
        self.embedding_net_text = embedding_net_text

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net_image(x1)
        output2 = self.embedding_net_text(x2)
        output3 = self.embedding_net_text(x3)
        return output1, output2, output3

    def get_embedding_image(self, x):
        return self.embedding_net_image(x)

    def get_embedding_text(self, x):
        return self.embedding_net_text(x)


class TripletNetText2Im(nn.Module):
    def __init__(self, embedding_net_image, embedding_net_text):
        super(TripletNetText2Im, self).__init__()
        self.embedding_net_image = embedding_net_image
        self.embedding_net_text = embedding_net_text

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net_text(x1)
        output2 = self.embedding_net_image(x2)
        output3 = self.embedding_net_image(x3)
        return output1, output2, output3

    def get_embedding_image(self, x):
        return self.embedding_net_image(x)

    def get_embedding_text(self, x):
        return self.embedding_net_text(x)
