import torch
from pytorch_lightning import LightningModule
from transformers import RobertaForSequenceClassification, RobertaTokenizer, AutoConfig
from scipy.special import softmax
import numpy as np

class TwitterSentimentAnalysis(LightningModule):
    def __init__(self, pretrained_model_name=f"cardiffnlp/twitter-roberta-base-sentiment-latest"):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(pretrained_model_name)
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.learning_rate = 2e-5
        
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)[0]

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        
        # Logging the loss
        self.log('train_loss', loss)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def loss_fn(self, logits, labels):
        return torch.nn.CrossEntropyLoss()(logits, labels)
    
    def train_epoch_end(self):
        # Logging the metrics at the end of each epoch
        self.log('train_loss', self.avg_loss)

if __name__ == "__main__":
    model = TwitterSentimentAnalysis()
    text = "I Love bjorn"
    encoded_input = model.tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0].detach().numpy()
    scores = softmax(scores)
    print(encoded_input)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    for i in range(scores.shape[0]):
        l = model.config.id2label[ranking[i]]
        s = scores[ranking[i]]
        print(f"{i+1}) {l} {np.round(float(s), 4)}")