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
df = pd.read_csv('test/, header=None)
df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
df
```
