import TrainingModel, Utils
import tensorflow as tf
import transformers
import numpy as np
import tqdm



def prepareDataset(tokenizer, numValidationSamples):
    # This part you need to prepare yourself!
    # What is needed here is a list of sentences in whatever language(s) you are interested in
    # and a matching set of Clip-Text encoder embeddings for the English counter part.

    # Pre-computed CLIP-Text encoder embeddings for 2 Million images, can be found here:
    # https://drive.google.com/drive/folders/1I9a7naSZubUATWzLFv61DQMWyFlF7wR5

    sentences = None
    emeddings = None
    print("Number of total training samples:", len(sentences))

    inSents, embs = shuffleData(sentences, emeddings)  # Shuffle before selecting validation data
    trainSents, trainEmbs = inSents[numValidationSamples:], embs[numValidationSamples:]
    evalSents, evalEmbs = inSents[:numValidationSamples], embs[:numValidationSamples]
    evalIds, evalAtt = Utils.batchEncode(evalSents, tokenizer)
    evalInData, evalLabels = (evalIds, evalAtt), tf.convert_to_tensor(evalEmbs, tf.float32)
    print("Number of training samples:", len(trainSents))
    print("Number of validation samples:", len(evalSents))

    return trainSents, trainEmbs, evalInData, evalLabels


def shuffleData(sents, embs):
    shuffleOrder = np.random.choice(range(len(sents)), len(sents), replace=False)
    f = lambda x: [x[i] for i in shuffleOrder]
    return f(sents), f(embs)


def createModel(modelBase, clipEmbeddingSize):
    model = TrainingModel.SentenceModelWithLinearTransformation(modelBase, clipEmbeddingSize)
    tokenizer = transformers.AutoTokenizer.from_pretrained(modelBase)
    return model, tokenizer


def trainStudentTextEncoder():
    modelBase = 'distilbert-base-multilingual-cased'
    numValidationSamples = 2000
    clipEmbeddingSize = 640
    learningRate = 5e-5
    batchSize = 64
    epochs = 100
    fetchSize = 500 * batchSize

    model, tokenizer = createModel(modelBase, clipEmbeddingSize)
    trainSents, trainEmbs, evalIn, evalLabels = prepareDataset(tokenizer, numValidationSamples)

    optim = tf.optimizers.Adam(learningRate)
    model.compile(optim, 'mse', metrics=['mae'])
    saveName = "CLIP-Text-Encoder"

    fetchCounter = 0
    for e in range(epochs):
        shuffleData(trainSents, trainEmbs)
        for i in tqdm.tqdm(range(0, len(trainSents), fetchSize), desc="Fetches"):
            batchEmbs = tf.convert_to_tensor(trainEmbs[i:i + fetchSize], tf.float32)
            batchSents = trainSents[i:i + fetchSize]

            inData = Utils.batchEncode(batchSents, tokenizer)

            model.fit(inData, batchEmbs, batch_size=batchSize, verbose=1,
                      validation_data=(evalIn, evalLabels), shuffle=True)

            fetchCounter += 1
            if (fetchCounter % 50 == 0):
                model.save_weights("{}-{}-Weights".format(saveName, fetchCounter))


if __name__ == '__main__':
    trainStudentTextEncoder()
