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


<!-- ABOUT THE PROJECT -->
## Overview
![Alt text](Images/Multilingual-CLIP.png?raw=true "Title")

[OpenAI](https://openai.com/) recently released the paper [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) in which they present the CLIP (Contrastive Language–Image Pre-training) model. This model is trained to connect text and images, by matching their corresponding vector representations using a contrastive learning objective.
CLIP consists of two separate models, a visual encoder and a text encoder. These were trained on a wooping 400 Million images and corresponding captions. 
OpenAI has since released a set of their smaller CLIP models, which can be found on the [official CLIP Github](https://github.com/openai/CLIP).

We propose a fine-tuning to replace the original English text encoder with a pre-trained text model in any language. This method makes it possible to adapt the powerful CLIP model to any language in roughly <b>24</b> GPU hours. <br>


#### This repository contains
* Pytorch inference code
* Tensorflow training code
* Pre-trained CLIP-Text encoders for multiple languages
* Training data and pre-computed CLIP text encodings for a large porton of the the image captions of [GCC](https://ai.google.com/research/ConceptualCaptions/) + [MSCOCO](https://cocodataset.org/#home) + [VizWiz](https://vizwiz.org/tasks-and-datasets/image-captioning/)

### Requirements
While it is possible that other versions works equally fine, we have worked with the following:

* Python = 3.6.9
* Transformers = 4.1.1
* Model Weights

## Usage
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
$ bash get-weights.sh
```

### Inference
```python
from src import multilingual_clip

print(multilingual_clip.AVAILABLE_MODELS.keys())

model = multilingual_clip.load_model('M-BERT-Distil-40')

embeddings = model(['Älgen är skogens konung!', 'Wie leben Eisbären in der Antarktis?', 'Вы знали, что все белые медведи левши?'])
print(embeddings.shape)
# Yields: torch.Size([3, 640])
```

<!--- For a more elaborative example see this [Google Colab](https://colab.research.google.com/github/FreddeFrallan/Multilingual-CLIP/blob/master/Multilingual_CLIP.ipynb). --->

For a more elaborate example, comparing the textual embeddings to the CLIP image embeddings see this [colab notebook](https://colab.research.google.com/github/FreddeFrallan/Multilingual-CLIP/blob/master/Multilingual_CLIP.ipynb).

<!-- GETTING STARTED -->
## Pre-trained Models
Every text encoder is a [Huggingface](https://huggingface.co/) available transformer, with an additional linear layer on top. Neither of the models have been extensively tested, but for more information and qualitative test results for a specific model, click the Model Name to see its model card.
<br>
<br>
<b>*** Make sure to update to the most recent version of the repostitory when downloading a new model, and re-run the shell script to download the Linear Weights. *** </b>

| Name |Model Base|Vision Model | Pre-trained Languages | Target Languages | #Parameters|
| ----------------------------------|:-----: |:-----: |:-----: |:-----: |:-----: |
|**Multilingual**    ||
| [M-BERT Distil 40](https://github.com/FreddeFrallan/Multilingual-CLIP/tree/main/Model%20Cards/M-BERT%20Distil%2040) | [M-BERT Distil](https://huggingface.co/bert-base-multilingual-uncased)|  RN50x4 | [101 Languages](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages) | [40 Languages](https://github.com/FreddeFrallan/Multilingual-CLIP/blob/main/Model%20Cards/M-BERT%20Distil%2040/Fine-Tune-Languages.md) | 66 M|
| [M-BERT Base 69](https://github.com/FreddeFrallan/Multilingual-CLIP/tree/main/Model%20Cards/M-BERT%20Base%2069) | [M-BERT Base](https://huggingface.co/bert-base-multilingual-uncased)|RN50x4 | [101 Languages](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages) | 68 Languages | 110 M|
|**Monolingual**    ||
|[Swe-CLIP 500k](https://github.com/FreddeFrallan/Multilingual-CLIP/tree/main/Model%20Cards/Swe-CLIP%20500k)| [KB-BERT](https://huggingface.co/KB/bert-base-swedish-cased)|  RN50x4 | Swedish | Swedish | 110 M|
|[Swe-CLIP 2M](https://github.com/FreddeFrallan/Multilingual-CLIP/tree/main/Model%20Cards/Swe-CLIP%202M)| [KB-BERT](https://huggingface.co/KB/bert-base-swedish-cased)|  RN50x4 | Swedish | Swedish | 110 M|

## Training a new model
[This folder](https://github.com/FreddeFrallan/Multilingual-CLIP/tree/main/src/TeacherLearning) contains the code used for training the above models. If you wsh to train your own model you must do the following things:

* Prepare a set of translated sentence pairs from English -> Your Language(s)
* Compute regular CLIP-Text embeddings for the English sentences.
* Edit [Training.py](https://github.com/FreddeFrallan/Multilingual-CLIP/blob/main/src/TeacherLearning/Training.py) to load your data.
* Train a new CLIP-Text encoder via Teacher Learning 

### Pre-computed CLIP Embeddings & Translaton Data
[This Google Drive folder]https://drive.google.com/drive/folders/1I9a7naSZubUATWzLFv61DQMWyFlF7wR5?usp=sharing) contains both pre-computed CLIP-Text Embeddings for a large porton of the the image captions of [GCC](https://ai.google.com/research/ConceptualCaptions/) + [MSCOCO](https://cocodataset.org/#home) + [VizWiz](https://vizwiz.org/tasks-and-datasets/image-captioning/).

The Google Drive folder also contains the translation data used to train the currently available models.
Good Luck

## Contribution
If you have trained a CLIP Text encoder specific to your language, or another model covering a language not supported here, Please feel free to contact us and we will either upload your model and credit you, or simply link to your already uploaded model.

<!-- CONTACT -->
## Contact
If you have questions regarding the code or otherwise related to this Github page, please open an [issue](https://github.com/FreddeFrallan/Contrastive-Tension/issues).

For other purposes, feel free to contact me directly at: Fredrk.Carlsson@ri.se

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
