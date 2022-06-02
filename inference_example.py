import transformers


def tf_example(texts, model_name='M-CLIP/XLM-Roberta-Large-Vit-L-14'):
    from multilingual_clip import tf_multilingual_clip

    model = tf_multilingual_clip.MultiLingualCLIP.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    inData = tokenizer.batch_encode_plus(texts, return_tensors='tf', padding=True)
    embeddings = model(inData)
    print(embeddings.shape)


def pt_example(texts, model_name='M-CLIP/XLM-Roberta-Large-Vit-L-14'):
    from multilingual_clip import pt_multilingual_clip

    model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    embeddings = model.forward(texts, tokenizer)
    print(embeddings.shape)


if __name__ == '__main__':
    exampleTexts = [
        'Three blind horses listening to Mozart.',
        'Älgen är skogens konung!',
        'Wie leben Eisbären in der Antarktis?',
        'Вы знали, что все белые медведи левши?'
    ]

    # tf_example(exampleTexts)
    pt_example(exampleTexts)
