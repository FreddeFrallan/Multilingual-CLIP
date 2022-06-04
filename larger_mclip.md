# Multilingual CLIP 2/6-2022

## Overview
Recently, OpenAI released some of their [bigger CLIP models](https://github.com/openai/CLIP/blob/main/model-card.md). Additionally, [OpenCLIP](https://github.com/mlfoundations/open_clip) is continuing to provide their large models, which have proven to match or even outperform the OpenAI models.

Thanks to the compute provided by [Stability.ai](https://stability.ai/) and [laion.ai](https://laion.ai/), we are now happy to announce that we provide multilingual text encoders for these models!
Along with:
 - Updated Inference & Training Code
 - The Corresponding Machine Translated Image Caption Dataset
 - PyPi package installer
 
 <br>
 
None of the M-CLIP models have been extensivly evaluated, but testing them on Txt2Img retrieval on the humanly translated MS-COCO dataset, we see the following **R@10** results:
| Name | En | De | Es | Fr | Zh | It | Pl | Ko | Ru | Tr | Jp |
| ----------------------------------|:-----: |:-----: |:-----: |:-----: | :-----: |:-----: |:-----: |:-----: |:-----: |:-----: |:-----: |
| [OpenAI CLIP Vit-B/32](https://github.com/openai/CLIP)| 90.3 | - | - | - | - | - | - | - | - | - | - |
| [OpenAI CLIP Vit-L/14](https://github.com/openai/CLIP)| 91.8 | - | - | - | - | - | - | - | - | - | - |
| [OpenCLIP ViT-B-16+-](https://github.com/openai/CLIP)| 94.3 | - | - | - | - | - | - | - | - | - | - |
| [LABSE Vit-L/14](https://huggingface.co/M-CLIP/LABSE-Vit-L-14)| 91.6 | 89.6 | 89.5 | 89.9 | 88.9 | 90.1 | 89.8 | 80.8 | 85.5 | 89.8 | 73.9 |
| [XLM-R Large Vit-B/32](https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-B-32)| 91.8 | 88.7 | 89.1 | 89.4 | 89.3 | 89.8| 91.4 | 82.1 | 86.1 | 88.8 | 81.0 |
| [XLM-R Vit-L/14](https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-L-14)| 92.4 | 90.6 | 91.0 | 90.0 | 89.7 | 91.1 | 91.3 | 85.2 | 85.8 | 90.3 | 81.9 |
| [XLM-R Large Vit-B/16+](https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-B-16Plus)| <b>95.0</b> | <b>93.0</b> | <b>93.6</b> | <b>93.1</b> | <b>94.0</b> | <b>93.1</b> | <b>94.4</b> | <b>89.0</b> | <b>90.0</b> | <b>93.0</b> | <b>84.2</b> |

To our surprise, using M-CLIP with XLM-RoBerta Large outperforms the original English models for English. Exactly why this is the case reamins to be determined, and we plan to followup up with more extensive testing.

The ViT-L/14 model is integrated into clip retrieval, you can test the retrieval capabilities of this multilingual encoder [there](https://rom1504.github.io/clip-retrieval/?useMclip=true&query=%E9%BB%84%E8%89%B2%E3%81%84%E7%8C%AB). This is a search over 5 billion of clip embeddings of laion5B dataset implemented with an efficient knn index.

The training curves for these models can be found at the [Weights and Biases report](https://wandb.ai/freddefrallan/M-CLIP/reports/M-CLIP-2-6-2022--VmlldzoyMTE1MjU1/edit?firstReport&runsetFilter)

## Training Data & Machine Translation
English image captions were taken from the Vit-L filtered captions of the datasets: [CC3M+CC12M+SBU](https://github.com/salesforce/BLIP#pre-training-datasets-download), which are provided by the BLIP repository.

From these 14 million captions we sampled 7 million captions, divided them into 48 equally sized buckets, and translated each bucket into one of the [48 target languages](https://github.com/FreddeFrallan/Multilingual-CLIP/blob/main/translation/data/fine_tune_languages.csv). This means that after translation we still end up with a total of 7 million captions. Where 7M/48 = 145,833 of them are in for example Dutch.
The machine-translated captions are available at [Huggingface](https://huggingface.co/datasets/M-CLIP/ImageCaptions-7M-Translations).

Each translation was performed with the corresponding Opus model. For more information see the [machine translation instructions](https://github.com/FreddeFrallan/Multilingual-CLIP/tree/main/translation).

It should be noted that only translated captions were used during training. Meaning that none of the original English captions were included. This entails that all the English (and other languages not included in the 49 target languages) results are due to transfer learning.

## Training Details
All released models used in essence the same hyperparameters. These detail are available at [Weights and Biases project](https://wandb.ai/freddefrallan/M-CLIP?workspace=user-freddefrallan).

Following is a short list of some of the shared hyperparameters:
 - Batch size of 2048 samples.
 - Adam Optimizer with a target learning rate of 10^-5, with a linear warmup schedule for 1k update steps.
 - 5000 randomly sampled validation samples

All models were allowed to train until the validation MSE loss had converged. For most models this took about 24 hours, using 8 Nvidia A-100 GPUs. No early stopping was performed in regard to the Image-Text retrieval tasks.

## Additional Experiments
In addition to the released models, we also performed some experiments that yielded negative or unsubstantial results. The training curves and specific settings for most of these additional experiments can be found at the [Weights and Biases project](https://wandb.ai/freddefrallan/M-CLIP?workspace=user-freddefrallan).

Following is a summary of things we tried:

- Optimizing the Cosine-Similarity instead of minimizing the mean-squared error: **No noticeable performance difference**.
 - MBERT-BASE as encoder: **Worse performance than LaBSE**
 - USE-CML: **Worse performance than LaBSE**
 - Adding additional TanH layer to the XLM-R Large: **No substantial performance difference, although it achieved slightly faster learning at the start.**
 - Using first *([CLS]?)* token as sentence embedding, instead of mean-pooling for XLM-R Large: **Significantly worse performance. *(Perhaps due to the lack of Next-Sentence Prediction task in the RoBerta pre-training?)***
