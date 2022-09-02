import pickle

import torch
import transformers


class MultilingualClip(torch.nn.Module):
    def __init__(self, model_name, tokenizer_name, head_name, weights_dir='data/weights/', cache_dir=None):
        super().__init__()
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.head_path = weights_dir + head_name

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
        self.transformer = transformers.AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
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


AVAILABLE_MODELS = {
    'M-BERT-Distil-40': {
        'model_name': 'M-CLIP/M-BERT-Distil-40',
        'tokenizer_name': 'M-CLIP/M-BERT-Distil-40',
        'head_name': 'M-BERT Distil 40 Linear Weights.pkl'
    },

    'M-BERT-Base-69': {
        'model_name': 'M-CLIP/M-BERT-Base-69',
        'tokenizer_name': 'M-CLIP/M-BERT-Base-69',
        'head_name': 'M-BERT-Base-69 Linear Weights.pkl'
    },

    'Swe-CLIP-500k': {
        'model_name': 'M-CLIP/Swedish-500k',
        'tokenizer_name': 'M-CLIP/Swedish-500k',
        'head_name': 'Swedish-500k Linear Weights.pkl'
    },

    'Swe-CLIP-2M': {
        'model_name': 'M-CLIP/Swedish-2M',
        'tokenizer_name': 'M-CLIP/Swedish-2M',
        'head_name': 'Swedish-2M Linear Weights.pkl'
    },
    
    'M-BERT-Base-ViT-B': {
        'model_name': 'M-CLIP/M-BERT-Base-ViT-B',
        'tokenizer_name': 'M-CLIP/M-BERT-Base-ViT-B',
        'head_name': 'M-BERT-Base-69-ViT Linear Weights.pkl'
    },
}


def load_model(name, cache_dir=None):
    config = AVAILABLE_MODELS[name]
    return MultilingualClip(**config, cache_dir=cache_dir)
