from src.multilingual_clip import MultilingualClip

model_path = 'M-CLIP/Swedish-500k'
tok_path = 'M-CLIP/Swedish-500k'
head_weight_path = 'data/weights/Swe-CLIP Linear Weights.pkl'

sweclip_args = {'model_name': model_path,
                'tokenizer_name': tok_path,
                'head_path': head_weight_path}

sweclip = MultilingualClip(**sweclip_args)

print(sweclip('test'))