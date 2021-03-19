<br />
<p align="center">
  <h1 align="center">M-BERT Distil 40</h1>
  
  <p align="center">  
    <a href="https://huggingface.co/M-CLIP/M-BERT-Distil-40">Huggingface Model</a>
    ·
    <a href="https://huggingface.co/distilbert-base-multilingual-cased">Huggingface Base Model</a>
  </p>
</p>


<!-- ABOUT THE PROJECT -->
## About
A[distilbert-base-multilingual](https://huggingface.co/distilbert-base-multilingual-cased) tuned to match the embedding space for 40 languages, to the embedding space of the CLIP text encoder which accomponies the Res50x4 vision encoder.
A full list of the languages which were used during fine-tuning can be found at [the end of this page](#supported-languages).

Training data pairs was generated by sampling 40k sentences for each language from the combined descriptions of [GCC](https://ai.google.com/research/ConceptualCaptions/) + [MSCOCO](https://cocodataset.org/#home) + [VizWiz](https://vizwiz.org/tasks-and-datasets/image-captioning/), and translating them into the corresponding language.
All translation was done using the [AWS translate service](https://aws.amazon.com/translate/), the quality of these translations have currently not been analyzed, but one can assume the quality varies between the 40 languages.


## Evaluation
#### French
![Alt](./French-Both.png)
#### German
![Alt](./German-Both.png)
#### Spanish
![Alt](./Spanish-Both.png)
#### Russian
![Alt](./Russian-Both.png)
#### Swedish
![Alt](./M-Swedish-Both.png)
#### Greek
![Alt](./Greek-Both.png)
#### Kannada
![Alt](./Kannada-Both.png)

(#supported-languages)