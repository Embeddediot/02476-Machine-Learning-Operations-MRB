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
        
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)[0]
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits.view(-1, 2), labels.view(-1))
        return {'loss': loss, 'log': {'train_loss': loss}}
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        val_loss = self.loss_fn(logits.view(-1, 2), labels.view(-1))
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        return {'val_loss': val_loss, 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        return {'val_loss': avg_loss, 'val_acc': avg_acc}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-5)

    def prepare_data(self, text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    
    def train_dataloader(self):
        # Return your twitter dataset dataloader here
        pass

    def val_dataloader(self):
        # Return your twitter dataset dataloader here
        pass






if __name__ == "__main__":
    model = TwitterSentimentAnalysis()
    text = "Covid cases are increasing fast!"
    text = model.prepare_data(text)
    encoded_input = model.tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0].detach().numpy()
    scores = softmax(scores)
    print(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    for i in range(scores.shape[0]):
        l = model.config.id2label[ranking[i]]
        s = scores[ranking[i]]
        print(f"{i+1}) {l} {np.round(float(s), 4)}")
