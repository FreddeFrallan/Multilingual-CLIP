from multilingual_clip import Config_MCLIP
import transformers
import torch


class MultilingualCLIP(transformers.PreTrainedModel):
    config_class = Config_MCLIP.MCLIPConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.transformer = transformers.AutoModel.from_pretrained(config.modelBase)
        self.LinearTransformation = torch.nn.Linear(in_features=config.transformerDimensions,
                                                    out_features=config.numDims)

    def forward(self, txt, tokenizer):
        txt_tok = tokenizer(txt, padding=True, return_tensors='pt')
        embs = self.transformer(**txt_tok)[0]
        att = txt_tok['attention_mask']
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return self.LinearTransformation(embs)

    @classmethod
    def _load_state_dict_into_model(cls, model, state_dict, pretrained_model_name_or_path, _fast_init=True):
        model.load_state_dict(state_dict)
        return model, [], [], []
