import TrainingModel
import transformers
import pickle


def convertTFTransformerToPT(saveNameBase):
    ptFormer = transformers.AutoModel.from_pretrained(saveNameBase + '-Transformer', from_tf=True)
    ptFormer.save_pretrained(saveNameBase + '-Transformer' + "-PT")

    with open('{}-Linear-Weights.pkl'.format(saveNameBase), 'rb') as fp:
        weights = pickle.load(fp)
    # TODO Add code for converting the linear weights into a torch linear layer


def splitAndStoreTFModelToDisk(transformerBase, weightsPath, visualDimensionSpace, saveNameBase):
    # Splits the Sentence Transformer and its linear layer
    # The Transformer can then be loaded into PT, and the linear weights can be added as a linear layer

    tokenizer = transformers.AutoTokenizer.from_pretrained(transformerBase)
    model = TrainingModel.SentenceModelWithLinearTransformation(transformerBase, visualDimensionSpace)
    model.load_weights(weightsPath).expect_partial()

    tokenizer.save_pretrained(saveNameBase + '-Tokenizer')
    model.transformer.save_pretrained(saveNameBase + '-Transformer')
    linearWeights = model.postTransformation.get_weights()
    print("Saving Linear Weights into pickle file.", linearWeights.shape)

    with open('{}-Linear-Weights.pkl'.format(saveNameBase), 'wb') as fp:
        pickle.dump(linearWeights, fp)


if __name__ == '__main__':
    weightsPath = 'XLM-Large-Sentence-VitB-16Plus-1652563598.5977607-135.weights'
    transformerBase = 'xlm-roberta-large'
    modelSaveBase = 'XLM-Large-VitB-16+'
    visualDimensionSpace = 640

    splitAndStoreTFModelToDisk(transformerBase, weightsPath, visualDimensionSpace, modelSaveBase)
    # convertTFTransformerToPT(modelSaveBase + "-Transformer")
