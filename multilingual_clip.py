import pickle

import torch
import transformers


class MultilingualClip(torch.nn.Module):
    def __init__(self, model_name, tokenizer_name, head_path):
        super().__init__()
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.head_path = head_path

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        self.transformer = transformers.AutoModel.from_pretrained(model_name)
        self.clip_head = torch.nn.Linear(in_features=768, out_features=640)
        self._load_head()

    def forward(self, txt):
        txt_tok = self.tokenizer(txt, padding=True, return_tensors='pt')
        embs = self.transformer(**txt_tok)[0]
        att = txt_tok['attention_mask']
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return self.clip_head(embs)

    def _load_head(self):
        with open(self.head_path, 'rb') as f:
            lin_weights = pickle.loads(f.read())
        self.clip_head.weight = torch.nn.Parameter(torch.tensor(lin_weights[0]).float().t())
        self.clip_head.bias = torch.nn.Parameter(torch.tensor(lin_weights[1]).float())