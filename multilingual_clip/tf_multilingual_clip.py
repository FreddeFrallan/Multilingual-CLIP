from multilingual_clip import Config_MCLIP
import tensorflow as tf
import transformers


class SentenceModel(tf.keras.Model):

    def __init__(self, modelBase, from_pt=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = transformers.TFAutoModel.from_pretrained(modelBase, from_pt=from_pt)

    @tf.function
    def generateMeanPooledSentenceEmbs(self, input, training=False):
        output = self.transformer(input, training=training)
        hiddenStates = output['last_hidden_state']

        outAtt = tf.cast(input['attention_mask'], tf.float32)
        sampleLength = tf.reduce_sum(outAtt, axis=-1, keepdims=True)
        maskedEmbs = hiddenStates * tf.expand_dims(outAtt, axis=-1)
        return tf.reduce_sum(maskedEmbs, axis=1) / tf.cast(sampleLength, tf.float32)

    @tf.function
    def call(self, inputs, training=False, mask=None):
        return self.generateMeanPooledSentenceEmbs(inputs, training)


class SentenceModelWithLinearTransformation(SentenceModel):

    def __init__(self, modelBase, embeddingSize=640, *args, **kwargs):
        super().__init__(modelBase, *args, **kwargs)
        self.postTransformation = tf.keras.layers.Dense(embeddingSize, activation='linear', name='LinearTransformation')

    @tf.function
    def call(self, inputs, training=False, mask=None):
        return self.postTransformation(self.generateMeanPooledSentenceEmbs(inputs, training))


class MultiLingualCLIP(transformers.TFPreTrainedModel):
    config_class = Config_MCLIP.MCLIPConfig

    @property
    def dummy_inputs(self):
        return {'input_ids': tf.ones((4, 12), tf.int32),
                'attention_mask': tf.ones((4, 12), tf.int32)}

    @tf.function(
        input_signature=[
            tf.TensorSpec((None, None), tf.int32), tf.TensorSpec((None, None), tf.int32)
        ]
    )
    def serving(self, ids, att):
        output = self.call((ids, att))
        return self.serving_output(output)

    def serving_output(self, outputs):
        return outputs

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.sentenceModel = SentenceModelWithLinearTransformation(config.modelBase, config.numDims)

    @tf.function
    def call(self, inputs, training=False, mask=None):
        return self.sentenceModel.call(inputs, training)
