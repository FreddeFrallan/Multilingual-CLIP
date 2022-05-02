A set of scripts to machine translate the subset of (synthetic) Conceptual Captions used in [BLIP](https://github.com/salesforce/BLIP#pre-training-datasets-download). The conda `environment.yml` file allows you to recreate the environment we used via `conda env create -f environment.yml` (creates env named `translate`).

## Step 0: Download data

```bash
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/ccs_synthetic_filtered_large.json -P data
```

## Step 1: Swedish captions 

Convert to tabular and save data to `.feather`. File is saved as `data/ccs_synthetic.feather`

```hash
python 01_ccs_to_df.py
```

Now translate captions from English -> Swedish. Will take about ~26 hours to translate 12.5 million captions with an RTX 3090.

```bash
python 01_translate_sv.py
```

### Step 2: Multilingual captions

We separate the original file into `n=150000` observations sized chunks for each language we are translating to. Run `02_multilingual_shard.py` to create a separate data file with 150k obs for each target language. Resulting files will be in `data_multi/` folder. The purpose of this script is also to merge target language metadata from `data/fine_tune_languages.csv` into the data files. This way the data files contain URL:s and languages codes for calling the correct language pair model and tokenizer names from `OPUS-MT` (via the `Helsinki-NLP` model repository on Huggingface). 

The list of language pairs and links to models were manually assembled. [This leaderboard](https://opus.nlpl.eu/leaderboard/) may help in getting an overview of further available pairs. 

```bash
python 02_multilingual_shard.py
``` 

Clean up and remove the chunks that did not have a target language (file names ending with `nan`)

```bash
find data_multi | grep nan.feather | xargs rm -f
```

Run the multilingual translation script for every data file. Use the bash script `02_translate_multi.sh`. Results will be stored in `data_translated/`.

```bash
bash 02_translate_multi.sh
```

### Step 3 (optional): Merge to single big file

Merge everything to the same file as the Swedish captions.

```bash
python 03_merge_sv_multi.py
```
