import pytorch_lightning as pl
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from transformers import RobertaForSequenceClassification, RobertaTokenizer, AutoConfig, BertTokenizer


train_dataset = pd.read_csv(r"./src/data/train.csv")

train_dataset = pd.read_csv(r"./src/data/train.csv")
val_dataset = pd.read_csv(r"./src/data/test.csv")
sample = pd.read_csv(r"./src/data/sample_submission.csv")


# #print(train_dataset['text'])



# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# # The DataLoader needs to know our batch size for training, so we specify it 
# # here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# # size of 16 or 32.
# batch_size = 32

# # Create the DataLoaders for our training and validation sets.
# # We'll take training samples in random order. 
# train_dataloader = DataLoader(
#             train_dataset,  # The training samples.
#             sampler = RandomSampler(train_dataset), # Select batches randomly
#             batch_size = batch_size # Trains with this batch size.
#         )

# # For validation the order doesn't matter, so we'll just read them sequentially.
# validation_dataloader = DataLoader(
#             val_dataset, # The validation samples.
#             sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
#             batch_size = batch_size # Evaluate with this batch size.
#         )

# print(validation_dataloader)




train_dataset = pd.read_csv(r"./src/data/train.csv")
val_dataset = pd.read_csv(r"./src/data/test.csv")
sample = pd.read_csv(r"./src/data/sample_submission.csv")



val_dataset = pd.read_csv(r"./src/data/test.csv")
sentences = val_dataset['text']
labels = val_dataset['sentiment']

labels = val_dataset['sentiment'].replace(['negative','neutral','positive'],[0,1,2])

#print(labels)

##Sentences: it is a list of of text data

##lables: is the label associated





###tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenizer = RobertaTokenizer.from_pretrained(f"cardiffnlp/twitter-roberta-base-sentiment-latest")



# Tokenize all of the sentences and map the tokens to thier word IDs.

input_ids = []
attention_mask = []

# For every sentence...

for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 100,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_mask.append(encoded_dict['attention_mask'])


# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_mask = torch.cat(attention_mask, dim=0)
labels = torch.tensor(labels)

# Print sentence 0, now as a list of IDs.
#print('Original: ', sentences[0])
#print('Token IDs:', input_ids[0])

### Not combine the input id , mask and labels and divide the dataset

#:
from torch.utils.data import TensorDataset, random_split

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_mask, labels)



# Create a 90-10 train-validation split.

# Calculate the number of samples to include in each set.
train_size = int(0.90 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#print('{:>5,} training samples'.format(train_size))
#print('{:>5,} validation samples'.format(val_size))

### Not you call loader of these datasets


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = 32

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )





































































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
        print("bruges 2")
        # Return your twitter dataset dataloader here
        return train_dataloader

    def val_dataloader(self):
        # Return your twitter dataset dataloader here
        print("bruges")
        return validation_dataloader



if __name__ == "__main__":
    model = TwitterSentimentAnalysis()
    text = "Covid cases are increasing fast!"
    text = model.prepare_data(text)
    encoded_input = model.tokenizer(text, return_tensors='pt')


    for data in dataset:

        #output = model(**encoded_input)
        output = model(data)
        scores = output[0].detach().numpy()
        scores = softmax(scores)
        print(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for i in range(scores.shape[0]):
            l = model.config.id2label[ranking[i]]
            s = scores[ranking[i]]
            print(f"{i+1}) {l} {np.round(float(s), 4)}")
