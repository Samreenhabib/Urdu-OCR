This repository contains a PyTorch implementation of the Urdu Printed Database dataset with the Hugging Face Transformers library. 
The Urdu Printed Database is a collection of handwritten English text images and their corresponding transcriptions.

****Installation****
To use the code in your repository, you'll need to install the following Python packages:

1. transformers
2. datasets
3. torchvision 
<br>
You can install these packages by running the following commands in your terminal or command prompt:

```
!pip install transformers
!pip install datasets
!pip install torchvision
```
Import the necessary classes from the transformers library:
```
from transformers import AutoFeatureExtractor, AutoTokenizer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
```
Load pre-trained feature extractor and tokenizer models from the transformers library:
```
feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-384")
decoder_tokenizer = AutoTokenizer.from_pretrained("urduhack/roberta-urdu-small")
```
Create a TrOCRProcessor object that combines the feature extractor and tokenizer models. 
The `processor.save_pretrained()` method saves the processor object to disk, so it can be loaded later. 
<br>
Import data from CSV file that contains the image filenames and the corresponding text that needs to be recognized. 
The rename() method renames the columns of the dataframe to "file_name" and "text".
```
import pandas as pd
import cv2
df = pd.read_csv('dataset/test.xlsx, header=None)
df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
df
```
Split the dataframe into a training set and a testing set. The train_test_split() method from the sklearn library is used to split the data. 
The training set will be used to train the OCR model, while the testing set will be used to evaluate its performance.
The reset_index() method is used to reset the indices of the dataframes so that they start from 0.
```
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.3)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
```
Instantiate the Dataset class for the training and validation sets using the following code:
```
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import TrOCRProcessor

class UrduDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, padding="max_length",max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

#instantiate TrOCRProcessor
processor = TrOCRProcessor.from_pretrained("./processor")

# instantiate training and validation datasets
train_dataset = UrduDataset(root_dir='./test/',
                           df=train_df,
                           processor=processor)
eval_dataset = UrduDataset(root_dir='./test/',
                          df=test_df,
                          processor=processor)

# print number of examples in datasets
print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))
```
Visualize a sample image from the dataset and check label:
```
# sample encoding from dataset
encoding = train_dataset[3]
for k,v in encoding.items():
    print(k, v.shape)

image = Image.open(train_dataset.root_dir + train_df['file_name'][5]).convert("RGB")
image
labels = encoding['labels']
labels[labels == -100] = processor.tokenizer.pad_token_id
label_str = processor.decode(labels, skip_special_tokens=True)
print(label_str)
```
Create a VisionEncoderDecoderModel from transformers:
```
from transformers import VisionEncoderDecoderModel
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained("google/vit-base-patch16-384", "urduhack/roberta-urdu-small")
# set decoder config to causal lm
model.config.decoder.is_decoder = True
model.config.decoder.add_cross_attention = True
# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size
# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4
```
Next, define some training hyperparameters by instantiating the `training_args`.
```
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    fp16=True, #uncomment this for CPU 
    output_dir="./",
    logging_steps=2,
    save_steps=1000,
    eval_steps=200,
    num_train_epochs=10,  
)
```
Now, evaluate the model on the Character Error Rate (CER), which is available in HuggingFace Datasets :
```
from datasets import load_metric
cer_metric = load_metric("cer")
```
The `compute_metrics` function takes an EvalPrediction input and returns a dictionary. 
During evaluation, the model outputs an EvalPrediction containing two items: 
*predictions
*ground-truth label ID
```
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}
 ```
 Let's train! 
```
from transformers import default_data_collator
# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator
)
trainer.train()
```
After complete training, save the trained model:
`trainer.save_model('./trainer')`
<br>
Now, call the model from pretrained directory:
`model = VisionEncoderDecoderModel.from_pretrained("./trainer")`
<br>
Test the model:
```
image = Image.open('/40.png').convert("RGB")
pixel_values = processor.feature_extractor(image, return_tensors="pt").pixel_values 
print(pixel_values.shape)
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
```

 

