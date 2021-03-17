<br />
<p align="center">
  <h1 align="center">Multilingual-CLIP</h1>
  <h3 align="center">OpenAI CLIP text encoders for multiple languages</h3>
  
  <p align="center">  
    <a href="https://openreview.net/pdf?id=Ov_sMNau-PF">Arxiv Paper</a>
    ·
    <a href="https://huggingface.co/Contrastive-Tension">Pre-trained Models</a>
    ·
    <a href="https://github.com/FreddeFrallan/Contrastive-Tension/issues">Report Bug</a>
  </p>
</p>


<!-- ABOUT THE PROJECT -->
## Overview
[OpenAI](https://openai.com/) recently released the paper [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) in which they present the CLIP model. This model is trained to connect text and images, by matching their corresponding vector representations using a contrastive learning objective.
CLIP consists of two separate models, a visual encoder and a text encoder. These were trained on a wooping 400 Million images and corresponding captions. 
OpenAI has since released a set of their smaller CLIP models, which can be found on the [official CLIP Github repo](https://github.com/openai/CLIP).

Since the existing vision and text model are separable, and since they have already done the hard task of creating a Vision-Text embedding space.
By leveraging already existing language models, one can easily fine-tune a new pre-trained text model to simply match the representations of the existing CLIP Text encoder. Hence, by utilizing machine translation and/or pre-trained multilingual models, one can cost-efficiently make CLIP available to a non-English language. For more details see our work-in-progress [Arxiv paper](www.google.com).


This repository contains:
* Pre-trained CLIP-Text encoders for multiple languages
* Tensorflow 2 and Pytorch inference code
* Tensorflow 2 code for training CLIP text encoders for any language
* ~3M pre-computed CLIP text encodings for the image captions of [GCC](https://ai.google.com/research/ConceptualCaptions/) + [MSCOCO](https://cocodataset.org/#home) + [VizWiz](https://vizwiz.org/tasks-and-datasets/image-captioning/)

### Requirements
While it is possible that other versions works equally fine, we have worked with the following:

* Python = 3.6.9
* Transformers = 4.1.1

<!-- USAGE EXAMPLES -->
## Usage
All the models and tokenizers are available via the Huggingface interface, and can be loaded for both Tensorflow and PyTorch:
```python
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained('Contrastive-Tension/RoBerta-Large-CT-STSb')

TF_model = transformers.TFAutoModel.from_pretrained('Contrastive-Tension/RoBerta-Large-CT-STSb')
PT_model = transformers.AutoModel.from_pretrained('Contrastive-Tension/RoBerta-Large-CT-STSb')
```

### Inference
To perform inference with the pre-trained models (or other Huggigface models) please see the script [ExampleBatchInference.py](ExampleBatchInference.py). <br>
The most important thing to remember when running inference is to apply the attention_masks on the batch output vector before mean pooling, as is done in the example script.

### CT Training
To run CT on your own models and text data see [ExampleTraining.py](ExampleTraining.py) for a comprehensive example. This file currently creates a dummy corpus of random text. Simply replace this to whatever corpus you like.

<!-- GETTING STARTED -->
## Pre-trained Models
Note that these models are <b>not</b> trained with the exact hyperparameters as those disclosed in the original CT paper. Rather, the parameters are from a short follow-up paper currently under review, which once again pushes the SOTA.

All evaluation is done using the [SentEval](https://github.com/facebookresearch/SentEval) framework, and shows the: (Pearson / Spearman) correlations
### Unsupervised / Zero-Shot
As both the training of BERT, and CT itself is fully self-supervised, the models only tuned with CT require no labeled data whatsoever.<br>
The NLI models however, are first fine-tuned towards a natural language inference task, which requires labeled data.

| Model| Avg Unsupervised STS |STS-b | #Parameters|
| ----------------------------------|:-----: |:-----: |:-----: |
|**Fully Unsupervised**    ||
| [BERT-Distil-CT](https://huggingface.co/Contrastive-Tension/BERT-Distil-CT)             | 75.12 / 75.04| 78.63 / 77.91 | 66 M|
| [BERT-Base-CT](https://huggingface.co/Contrastive-Tension/BERT-Base-CT)  | 73.55 / 73.36 | 75.49 / 73.31 | 108 M|
| [BERT-Large-CT](https://huggingface.co/Contrastive-Tension/BERT-Large-CT)        | 77.12 / 76.93| 80.75 / 79.82 | 334 M|
|**Using NLI Data**    ||
| [BERT-Distil-NLI-CT](https://huggingface.co/Contrastive-Tension/BERT-Distil-NLI-CT)             | 76.65 / 76.63 | 79.74 / 81.01 | 66 M|
| [BERT-Base-NLI-CT](https://huggingface.co/Contrastive-Tension/BERT-Base-NLI-CT)  | 76.05 / 76.28 | 79.98 / 81.47  | 108 M|
| [BERT-Large-NLI-CT](https://huggingface.co/Contrastive-Tension/BERT-Large-NLI-CT)        | <b> 77.42 / 77.41 </b> | <b> 80.92 / 81.66 </b>  | 334 M|

### Supervised
These models are fine-tuned directly with STS data, using a modified version of the supervised training object proposed by [S-BERT](https://arxiv.org/abs/1908.10084).<br>
To our knowledge our RoBerta-Large-STSb is the current SOTA model for STS via sentence embeddings.

| Model| STS-b | #Parameters|
| ----------------------------------|:-----: |:-----: |
| [BERT-Distil-CT-STSb](https://huggingface.co/Contrastive-Tension/BERT-Distil-CT-STSb)             | 84.85 / 85.46  | 66 M|
| [BERT-Base-CT-STSb](https://huggingface.co/Contrastive-Tension/BERT-Base-CT-STSb)  | 85.31 / 85.76  | 108 M|
| [BERT-Large-CT-STSb](https://huggingface.co/Contrastive-Tension/BERT-Large-CT-STSb)        | 85.86 / 86.47  | 334 M|
| [RoBerta-Large-CT-STSb](https://huggingface.co/Contrastive-Tension/RoBerta-Large-CT-STSb)        | <b> 87.56 / 88.42 </b>  | 334 M|

### Other Languages

| Model | Language | #Parameters|
| ----------------------------------|:-----: |:-----: |
| [BERT-Base-Swe-CT-STSb](https://huggingface.co/Contrastive-Tension/BERT-Base-Swe-CT-STSb/tree/main)             | Swedish  | 108 M|



<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact
If you have questions regarding the paper, please consider creating a comment via the official [OpenReview submission](https://openreview.net/forum?id=Ov_sMNau-PF). </br>
If you have questions regarding the code or otherwise related to this Github page, please open an [issue](https://github.com/FreddeFrallan/Contrastive-Tension/issues).

For other purposes, feel free to contact me directly at: Fredrk.Carlsson@ri.se

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [SentEval](https://github.com/facebookresearch/SentEval)
* [Huggingface](https://huggingface.co/)
* [Sentence-Transformer](https://github.com/UKPLab/sentence-transformers)
* [Best Readme Template](https://github.com/othneildrew/Best-README-Template)


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
