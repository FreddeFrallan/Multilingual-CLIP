# Multilingual CLIP Training 2/6-2022

## Training Data & Machine Translation
English image captions were taken from the Vit-L filtered captions of the datasets: [CC3M+CC12M+SBU](https://github.com/salesforce/BLIP#pre-training-datasets-download), which are provided by the BLIP repostitory. 

From these 14 million captions we sampled 7 million captions, divded them into 48 equally sized buckets, and translated each bucket into one of the [48 target languages](https://github.com/FreddeFrallan/Multilingual-CLIP/blob/main/translation/data/fine_tune_languages.csv). This means that after translation we still up with a total of 7 million captions. Where 7M/48 = 145,833 of them are in for example Dutch.

Each translation was performed with the corresponding Opus model. For more information see the [machine translation instructions](https://github.com/FreddeFrallan/Multilingual-CLIP/tree/main/translation).

It should be noted that only translated captions were used during training. Meaning that none of the original English captions were included. This entails that all the English (and other languages not included in the 49 target languages) results are due to transfer learning.

## Training Details
All released models used in essence the same hyperparameters. These details, along with the corresponding training curves are available at [Weights and Biases](https://wandb.ai/freddefrallan/Multilingual-CLIP).

Following is short list of some of the shared hyperparameters:
 - Batch size of 2048 samples.
 - Adam Optimizer with a target learning rate of 10^-5, with a linear warmup schedule for 1k update steps.
 - 5000 randomly sampled validation samples

All models were allowed to train until the validation mse loss had converged. For most models this took about 24 hours, using 8 Nvidia A-100 GPUs. No early stopping was performed in regards to the Image-Text retrieval tasks.

## Evaluation results

see https://github.com/FreddeFrallan/Multilingual-CLIP#validation--training-curves

## Additional Experiments
In addition to the released models we also performed some experiments that yielded negative, or unsubstantal results. The training curves and specific settings for most of these additional experiments can be found at the [Additional weights and biases](https://wandb.ai/freddefrallan/M-CLIP?workspace=user-freddefrallan).

Following is a summary of things we tried:

- Optimizing the Cosine-Similarity instead of minimizing the mean-squared error: **No noticeable performance difference**.
 - MBERT-BASE as encoder: **Worse performance than LaBSE**
 - USE-CML: **Worse performance than LaBSE**
 - Adding additional TanH layer to the XLM-R Large: **No substantial performance difference, although it achieved slightly faster learning in the start.**
 - Using first *([CLS]?)* token as sentence embedding, instead of mean-pooling for XLM-R Large: **Significantly worse performance. *(Perhaps due to the lack of Next-Sentence Prediction task in the RoBerta pre-training?)***
