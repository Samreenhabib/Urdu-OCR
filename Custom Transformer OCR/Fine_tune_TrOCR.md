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
split the dataframe into a training set and a testing set. The train_test_split() method from the sklearn library is used to split the data. 
The training set will be used to train the OCR model, while the testing set will be used to evaluate its performance.
The reset_index() method is used to reset the indices of the dataframes so that they start from 0.
```
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.3)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
```
