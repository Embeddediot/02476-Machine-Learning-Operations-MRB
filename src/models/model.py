import torch
from pytorch_lightning import LightningModule
from transformers import RobertaForSequenceClassification, RobertaTokenizer

class TwitterSentimentAnalysis(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("roberta-base")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

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
    
    def train_dataloader(self):
        # Return your twitter dataset dataloader here
        pass

    def val_dataloader(self):
        # Return your twitter dataset dataloader here
        pass