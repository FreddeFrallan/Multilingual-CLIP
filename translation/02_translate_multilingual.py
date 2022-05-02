import os
import pandas as pd
import torch
import argparse
import shutil
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", type=str)
parser.add_argument(
    "data_folder",
    nargs="?",
    type=str,
    default="data_multi",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("data_translated", exist_ok=True)

if args.filename == "015_ccs_synthetic_en.feather":
    # No need to translate English -> English
    shutil.copy2(os.path.join(args.data_folder, "015_ccs_synthetic_en.feather"), "data_translated")
    os._exit(0)

df = pd.read_feather(os.path.join(args.data_folder, args.filename))
df["opus_mt_url"] = df["opus_mt_url"].str.replace("https://huggingface.co/", "")
print(f"Starting translation of English to {df['multi_language_name'][0]}")


class CaptionDataset(Dataset):
    def __init__(self, df, tokenizer_name):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sentence1 = df.loc[index, "caption"]

        tokens = self.tokenizer(sentence1, return_tensors="pt")

        return tokens


tokenizer = AutoTokenizer.from_pretrained(df["opus_mt_url"][0])
model = AutoModelForSeq2SeqLM.from_pretrained(df["opus_mt_url"][0])
model.to(device)
model.eval()


def custom_collate_fn(data):
    """
    Data collator with padding.
    """
    tokens = [sample["input_ids"][0] for sample in data]
    attention_masks = [sample["attention_mask"][0] for sample in data]

    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True)
    padded_tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)

    batch = {"input_ids": padded_tokens, "attention_mask": attention_masks}
    return batch


if df["multi_target"][0] == 1:
    # If model is a multilingual model we need to concatenate target language code
    # in the form '>> CODE >>' in front of string so model outputs correct language.
    df["caption"] = ">>" + df["target_code"] + "<<" + df["caption"]

test_data = CaptionDataset(df, df["opus_mt_url"][0])
test_dataloader = DataLoader(
    test_data,
    batch_size=50,
    shuffle=False,
    num_workers=4,
    collate_fn=custom_collate_fn,
)

with torch.no_grad():
    decoded_tokens = []
    for i, batch in enumerate(tqdm(test_dataloader)):

        batch = {k: v.to(device) for k, v in batch.items()}
        output_tokens = model.generate(**batch)
        decoded_tokens += tokenizer.batch_decode(output_tokens.to("cpu"), skip_special_tokens=True)

df["caption_multi"] = decoded_tokens
df.to_feather(os.path.join("data_translated", args.filename))

print(f"Finished translating English to {df['multi_language_name'][0]}")
