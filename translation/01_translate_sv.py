import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader

df = pd.read_feather("data/ccs_synthetic.feather")

# from transformers import MarianTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-sv")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-sv")
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


test_data = CaptionDataset(df, "Helsinki-NLP/opus-mt-en-sv")
test_dataloader = DataLoader(
    test_data,
    batch_size=64,
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


df["caption_sv"] = decoded_tokens
df.to_feather("data/ccs_synthetic_sv.feather")
