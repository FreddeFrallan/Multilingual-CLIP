<br />
<p align="center">
  <h1 align="center">Multilingual-CLIP</h1>
  <h3 align="center">OpenAI CLIP text encoders for any language</h3>
  
  <p align="center">  
    <a href="https://colab.research.google.com/github/FreddeFrallan/Multilingual-CLIP/blob/master/Multilingual_CLIP.ipynb">Colab Notebook</a>
    ·
    <a href="https://huggingface.co/M-CLIP">Pre-trained Models</a>
    ·
    <a href="https://github.com/FreddeFrallan/Contrastive-Tension/issues">Report Bug</a>
  </p>
</p>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FreddeFrallan/Multilingual-CLIP/blob/master/Multilingual_CLIP.ipynb)
[![pypi](https://img.shields.io/pypi/v/multilingual-clip.svg)](https://pypi.python.org/pypi/multilingual-clip)


<!-- ABOUT THE PROJECT -->
## Overview
![Alt text](Images/Multilingual-CLIP.png?raw=true "Title")

[OpenAI](https://openai.com/) recently released the paper [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) in which they present the CLIP (Contrastive Language–Image Pre-training) model. This model is trained to connect text and images, by matching their corresponding vector representations using a contrastive learning objective.
CLIP consists of two separate models, a visual encoder and a text encoder. These were trained on a wooping 400 Million images and corresponding captions. 
OpenAI has since released a set of their smaller CLIP models, which can be found on the [official CLIP Github](https://github.com/openai/CLIP).


#### This repository contains
* Pre-trained CLIP-Text encoders for multiple languages
* Pytorch & Tensorflow inference code
* Tensorflow training code

### Requirements
While it is possible that other versions works equally fine, we have worked with the following:

* Python = 3.6.9
* Transformers = 4.8.1

## Install

`pip install multilingual-clip torch`

You can also choose to `pip install tensorflow` instead of torch.


## Inference Usage

Inference code for Tensorflow is also available in [inference_example.py](https://github.com/FreddeFrallan/Multilingual-CLIP/blob/main/inference_example.py)

```python
from multilingual_clip import pt_multilingual_clip
import transformers

texts = [
    'Three blind horses listening to Mozart.',
    'Älgen är skogens konung!',
    'Wie leben Eisbären in der Antarktis?',
    'Вы знали, что все белые медведи левши?'
]
model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'

# Load Model & Tokenizer
model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

embeddings = model.forward(texts, tokenizer)
print(embeddings.shape)
```

## Install for development

Setup a virtualenv:

```
python3 -m venv .env
source .env/bin/activate
pip install -e .
```

## Pre-trained Models
Every text encoder is a [Huggingface](https://huggingface.co/) available transformer, with an additional linear layer on top. For more information of a specific model, click the Model Name to see its model card.
<br>
<br>

| Name |Model Base|Vision Model | Vision Dimensions | Pre-trained Languages | #Parameters|
| ----------------------------------|:-----: |:-----: |:-----: |:-----: | :-----: |
| [LABSE Vit-L/14](https://huggingface.co/M-CLIP/LABSE-Vit-L-14)| [LaBSE](https://huggingface.co/sentence-transformers/LaBSE)|  [OpenAI ViT-L/14](https://github.com/openai/CLIP) | 768 | [109 Languages](https://arxiv.org/pdf/2007.01852.pdf) | 110 M|
| [XLM-R Large Vit-B/32](https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-B-32)| [XLM-Roberta-Large](https://huggingface.co/xlm-roberta-large)|  [OpenAI ViT-B/32](https://github.com/openai/CLIP) | 512 | [100 Languages](https://github.com/facebookresearch/fairseq/tree/main/examples/xlmr#Introduction) | 344 M|
| [XLM-R Large Vit-L/14](https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-L-14)| [XLM-Roberta-Large](https://huggingface.co/xlm-roberta-large)|  [OpenAI ViT-L/14](https://github.com/openai/CLIP) | 768 | [100 Languages](https://github.com/facebookresearch/fairseq/tree/main/examples/xlmr#Introduction)|  344 M|
| [XLM-R Large Vit-B/16+](https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-B-16Plus)| [XLM-Roberta-Large](https://huggingface.co/xlm-roberta-large)|  [Open CLIP ViT-B-16-plus-240](https://github.com/mlfoundations/open_clip) | 640 | [100 Languages](https://github.com/facebookresearch/fairseq/tree/main/examples/xlmr#Introduction)| 344 M|

### Validation & Training Curves
Following is a table of the <b>Txt2Img @10-Recal</b> for the humanly tanslated [MS-COCO testset](https://arxiv.org/abs/2109.07622).

| Name | En | De | Es | Fr | Zh | It | Pl | Ko | Ru | Tr | Jp |
| ----------------------------------|:-----: |:-----: |:-----: |:-----: | :-----: |:-----: |:-----: |:-----: |:-----: |:-----: |:-----: |
| [OpenAI CLIP Vit-B/32](https://github.com/openai/CLIP)| 90.3 | - | - | - | - | - | - | - | - | - | - |
| [OpenAI CLIP Vit-L/14](https://github.com/openai/CLIP)| 91.8 | - | - | - | - | - | - | - | - | - | - |
| [OpenCLIP ViT-B-16+-](https://github.com/openai/CLIP)| 94.3 | - | - | - | - | - | - | - | - | - | - |
| [LABSE Vit-L/14](https://huggingface.co/M-CLIP/LABSE-Vit-L-14)| 91.6 | 89.6 | 89.5 | 89.9 | 88.9 | 90.1 | 89.8 | 80.8 | 85.5 | 89.8 | 73.9 |
| [XLM-R Large Vit-B/32](https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-B-32)| 91.8 | 88.7 | 89.1 | 89.4 | 89.3 | 89.8| 91.4 | 82.1 | 86.1 | 88.8 | 81.0 |
| [XLM-R Vit-L/14](https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-L-14)| 92.4 | 90.6 | 91.0 | 90.0 | 89.7 | 91.1 | 91.3 | 85.2 | 85.8 | 90.3 | 81.9 |
| [XLM-R Large Vit-B/16+](https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-B-16Plus)| <b>95.0</b> | <b>93.0</b> | <b>93.6</b> | <b>93.1</b> | <b>94.0</b> | <b>93.1</b> | <b>94.4</b> | <b>89.0</b> | <b>90.0</b> | <b>93.0</b> | <b>84.2</b> |

The training curves for these models are available at this [Weights and Biases](https://wandb.ai/freddefrallan/Multilingual-CLIP?workspace=user-freddefrallan).

## Legacy Usage and Models
Older versions of M-CLIP had the linear weights stored separately from Huggingface. Whilst the new models have them directly incorporated in the Huggingface repository. More information about these older models can be found in this section. 

<details>
  <summary>Click for more information</summary>
  
##### Download CLIP Model
```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```
Replace `cudatoolkit=11.0` above with the appropriate CUDA version on your machine or `cpuonly` when installing on a machine without a GPU.
For more information please see the official [CLIP repostitory](https://github.com/openai/CLIP).
##### Download Linear Weights
```bash
# Linear Model Weights
$ bash legacy_get-weights.sh
```

### Inference
```python
from multilingual_clip import multilingual_clip

print(multilingual_clip.AVAILABLE_MODELS.keys())

model = multilingual_clip.load_model('M-BERT-Distil-40')

embeddings = model(['Älgen är skogens konung!', 'Wie leben Eisbären in der Antarktis?', 'Вы знали, что все белые медведи левши?'])
print(embeddings.shape)
# Yields: torch.Size([3, 640])
```

<!--- For a more elaborative example see this [Google Colab](https://colab.research.google.com/github/FreddeFrallan/Multilingual-CLIP/blob/master/Multilingual_CLIP.ipynb). --->

For a more elaborate example, comparing the textual embeddings to the CLIP image embeddings see this [colab notebook](https://colab.research.google.com/github/FreddeFrallan/Multilingual-CLIP/blob/master/Multilingual_CLIP.ipynb).

<!-- GETTING STARTED -->
## Legacy Pre-trained Models
Every text encoder is a [Huggingface](https://huggingface.co/) available transformer, with an additional linear layer on top. Neither of the models have been extensively tested, but for more information and qualitative test results for a specific model, click the Model Name to see its model card.
<br>
<br>
<b>*** Make sure to update to the most recent version of the repostitory when downloading a new model, and re-run the shell script to download the Linear Weights. *** </b>


| Name |Model Base|Vision Model | Pre-trained Languages | Target Languages | #Parameters|
| ----------------------------------|:-----: |:-----: |:-----: |:-----: |:-----: |
|**Multilingual**    ||
| [M-BERT Distil 40](https://github.com/FreddeFrallan/Multilingual-CLIP/tree/main/Model%20Cards/M-BERT%20Distil%2040) | [M-BERT Distil](https://huggingface.co/bert-base-multilingual-uncased)|  RN50x4 | [101 Languages](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages) | [40 Languages](https://github.com/FreddeFrallan/Multilingual-CLIP/blob/main/Model%20Cards/M-BERT%20Distil%2040/Fine-Tune-Languages.md) | 66 M|
| [M-BERT Base 69](https://github.com/FreddeFrallan/Multilingual-CLIP/tree/main/Model%20Cards/M-BERT%20Base%2069) | [M-BERT Base](https://huggingface.co/bert-base-multilingual-uncased)|RN50x4 | [101 Languages](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages) | 68 Languages | 110 M|
| [M-BERT Base ViT-B](https://github.com/FreddeFrallan/Multilingual-CLIP/tree/main/Model%20Cards/M-BERT%20Base%20ViT-B) | [M-BERT Base](https://huggingface.co/bert-base-multilingual-uncased)|ViT-B/32 | [101 Languages](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages) | 68 Languages | 110 M|
|**Monolingual**    ||
|[Swe-CLIP 500k](https://github.com/FreddeFrallan/Multilingual-CLIP/tree/main/Model%20Cards/Swe-CLIP%20500k)| [KB-BERT](https://huggingface.co/KB/bert-base-swedish-cased)|  RN50x4 | Swedish | Swedish | 110 M|
|[Swe-CLIP 2M](https://github.com/FreddeFrallan/Multilingual-CLIP/tree/main/Model%20Cards/Swe-CLIP%202M)| [KB-BERT](https://huggingface.co/KB/bert-base-swedish-cased)|  RN50x4 | Swedish | Swedish | 110 M|

  </details>
  
## Training a new model
[This folder](https://github.com/FreddeFrallan/Multilingual-CLIP/tree/main/multilingual_clip/TeacherLearning) contains the code used for training the above models. If you wsh to train your own model you must do the following things:

* Prepare a set of translated sentence pairs from English -> Your Language(s)
* Compute regular CLIP-Text embeddings for the English sentences.
* Edit [Training.py](https://github.com/FreddeFrallan/Multilingual-CLIP/blob/main/multilingual_clip/TeacherLearning/Training.py) to load your data.
* Train a new CLIP-Text encoder via Teacher Learning 

### Pre-computed CLIP Embeddings & Translaton Data
[This Google Drive folder](https://drive.google.com/drive/folders/1I9a7naSZubUATWzLFv61DQMWyFlF7wR5?usp=sharing) contains both pre-computed CLIP-Text Embeddings for a large porton of the the image captions of [GCC](https://ai.google.com/research/ConceptualCaptions/) + [MSCOCO](https://cocodataset.org/#home) + [VizWiz](https://vizwiz.org/tasks-and-datasets/image-captioning/).

The Google Drive folder also contains the translation data used to train the currently available models.
Good Luck

## Contribution
If you have trained a CLIP Text encoder specific to your language, or another model covering a language not supported here, Please feel free to contact us and we will either upload your model and credit you, or simply link to your already uploaded model.

<!-- CONTACT -->
## Contact
If you have questions regarding the code or otherwise related to this Github page, please open an [issue](https://github.com/FreddeFrallan/Contrastive-Tension/issues).

For other purposes, feel free to contact me directly at: Fredrik.Carlsson@ri.se

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [CLIP](https://openai.com/blog/clip/)
* [OpenAI](https://openai.com/)
* [Huggingface](https://huggingface.co/)
* [Best Readme Template](https://github.com/othneildrew/Best-README-Template)
* ["Two Cats" Image by pl1602](https://search.creativecommons.org/photos/8dfd802b-58e5-4cc5-889d-96abba540de1)

<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE` for more information.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
