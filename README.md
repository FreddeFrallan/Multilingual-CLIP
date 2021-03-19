<br />
<p align="center">
  <h1 align="center">Multilingual-CLIP</h1>
  <h3 align="center">OpenAI CLIP text encoders for any language</h3>
  
  <p align="center">  
    <a href="https://colab.research.google.com/github/FreddeFrallan/Multilingual-CLIP/blob/master/Multilingual_CLIP.ipynb">Colab Demo</a>
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
* Pre-trained CLIP-Text encoders for multiple languages
* Training data and ~3M pre-computed CLIP text encodings for the image captions of [GCC](https://ai.google.com/research/ConceptualCaptions/) + [MSCOCO](https://cocodataset.org/#home) + [VizWiz](https://vizwiz.org/tasks-and-datasets/image-captioning/)

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
from src.multilingual_clip import MultilingualClip

model_path = 'M-CLIP/M-BERT-Distil-40'
tok_path = 'M-CLIP/M-BERT-Distil-40'
head_weight_path = 'data/weights/M-BERT Distil 40 Linear Weights.pkl'

sweclip_args = {'model_name': model_path,
                'tokenizer_name': tok_path,
                'head_path': head_weight_path}

sweclip = MultilingualClip(**sweclip_args)

print(sweclip('test'))
```

For a more elaborative example see this [Google Colab](https://colab.research.google.com/github/FreddeFrallan/Multilingual-CLIP/blob/master/Multilingual_CLIP.ipynb).

<!-- GETTING STARTED -->
## Pre-trained Models
Every text encoder is a [Huggingface](https://huggingface.co/) available transformer, with an additional linear layer on top, available at [GoogleDrive](www.google.drive.com). We recommend downloading them seperatly to not struggre with Tensorflow/PyTorch versions. But for conveniance, the transformer and the linear layer can also be downloaded as a complete Tensorflow/PyTorch model directly from GoogleDrive aswell. <br> 

| Name |Model Base| Pre-trained Languages | Target Languages | #Parameters|
| ----------------------------------|:-----: |:-----: |:-----: |:-----: |
|**Multilingual**    ||
|| [M-BERT Distil](https://huggingface.co/bert-base-multilingual-uncased)             | 101 Languages | 68 Languages | 66 M|
| [M-BERT Distil 40](https://huggingface.co/M-CLIP/M-BERT-Distil-40) | [M-BERT Distil](https://huggingface.co/bert-base-multilingual-uncased)             | 101 Languages | 40 Languages | 66 M|
| | [M-BERT Base](https://huggingface.co/bert-base-multilingual-uncased)             | 101 Languages | 68 Languages | 110 M|
|**Single**    ||
|[Swe-CLIP 500k](https://huggingface.co/M-CLIP/Swedish-500k)| [KB-BERT](https://huggingface.co/bert-base-multilingual-uncased)             | Swedish | Swedish | 110 M|

## Contribution
If you have trained a CLIP Text encoder specific to your language, or another model covering a language not supported here, Please feel free to contact us and we will either upload your model and credit you, or simply link to your already uploaded model.

<!-- CONTACT -->
## Contact
If you have questions regarding the code or otherwise related to this Github page, please open an [issue](https://github.com/FreddeFrallan/Contrastive-Tension/issues).

For other purposes, feel free to contact me directly at: Fredrk.Carlsson@ri.se

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [CLIP] (https://openai.com/blog/clip/)
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
