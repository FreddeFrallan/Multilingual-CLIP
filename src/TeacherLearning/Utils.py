def batchEncode(texts, tokenizer, maxSentLen=128):
    inData = tokenizer.batch_encode_plus(texts, add_special_tokens=True, max_length=maxSentLen,
                                         truncation=True, padding=True, return_tensors='tf')

    return inData['input_ids'], inData['attention_mask']
