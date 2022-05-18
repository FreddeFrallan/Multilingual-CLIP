from tensorflow_addons.utils import types
from typeguard import typechecked
import tensorflow as tf
import numpy as np
import pickle

def splitListIntoChunks(data, numChunks):
    chunkSize = int(len(data) / numChunks)
    chunks = []
    for i in range(numChunks - 1):
        start, end = i * chunkSize, (i + 1) * chunkSize
        chunks.append(data[start:end])

    chunks.append(data[end:])
    return chunks


def splitIntoValueChunks(data, numChunks, getValueFunc):
    values = [getValueFunc(d) for d in data]
    minValue, maxValue = np.min(values), np.max(values)
    chunkSize = (maxValue - minValue) / float(numChunks)

    data.sort(key=lambda x: getValueFunc(x))
    sizeCeil = minValue + chunkSize
    chunks, currentChunkIndex = [[]], 0
    for d in data:
        v = getValueFunc(d)
        while (v > sizeCeil):
            chunks.append([])
            sizeCeil += chunkSize
            currentChunkIndex += 1
        chunks[currentChunkIndex].append(d)

    return chunks


def startGraphLogging():
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = 'logs/func/%s' % stamp
    writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on(graph=True, profiler=True)
    return writer, logdir


def finishGraphLogging(writer, logdir):
    with writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace",
            step=0,
            profiler_outdir=logdir)


class CustomSaveCallBack(tf.keras.callbacks.Callback):

    def __init__(self, saveName, saveInterval=10, firstSavePoint=-1):
        super().__init__()
        self.saveName = saveName
        self.saveInterval = saveInterval
        self.firstSavePoint = saveInterval if firstSavePoint < 0 else firstSavePoint
        self.saveCounter = 0

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1 >= self.firstSavePoint):
            if (self.saveCounter % self.saveInterval == 0):
                print("Saving model!")
                self.model.save_weights(self.saveName.format(epoch + 1))

            self.saveCounter += 1


def saveTokenizer(base='gpt2', dumpPath='GPT2-Tokenizer.pkl'):
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(base)
    with open(dumpPath, 'wb') as fp:
        pickle.dump(tokenizer, fp)


def loadTokenizer(dumpPath='GPT2-Tokenizer.pkl'):
    with open(dumpPath, 'rb') as fp:
        return pickle.load(fp)


class GradientAccumulator(tf.keras.optimizers.Optimizer):
    """Optimizer wrapper for gradient accumulation."""

    @typechecked
    def __init__(
            self,
            inner_optimizer: types.Optimizer,
            accum_steps: types.TensorLike = 4,
            name: str = "GradientAccumulator",
            **kwargs,
    ):
        r"""Construct a new GradientAccumulator optimizer.
        Args:
            inner_optimizer: str or `tf.keras.optimizers.Optimizer` that will be
                used to compute and apply gradients.
            accum_steps: int > 0. Update gradient in every accumulation steps.
            name: Optional name for the operations created when applying
                gradients. Defaults to "GradientAccumulator".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
                norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(name, **kwargs)
        self._optimizer = tf.keras.optimizers.get(inner_optimizer)
        self._gradients = []
        self._accum_steps = accum_steps
        self._step = None
        self._iterations = self._optimizer.iterations

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)
        for var in var_list:
            self.add_slot(var, "ga")

        self._gradients = [self.get_slot(var, "ga") for var in var_list]

    @property
    def step(self):
        """Variable. The number of training steps this Optimizer has run."""
        if self._step is None:
            with self._distribution_strategy_scope():
                self._step = self.add_weight(
                    "iter",
                    shape=[],
                    initializer="ones",
                    dtype=tf.int64,
                    trainable=False,
                    aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                )
            self._weights.append(self._step)
        return self._step

    @step.setter
    def step(self, variable):
        if self._step is not None:
            raise RuntimeError(
                "Cannot set `step` to a new Variable after "
                "the Optimizer weights have been created"
            )
        self._step = variable
        self._weights.append(self._step)

    @property
    def gradients(self):
        """The accumulated gradients on the current replica."""
        if not self._gradients:
            raise ValueError(
                "The accumulator should be called first to initialize the gradients"
            )
        return list(
            gradient.read_value() if gradient is not None else gradient
            for gradient in self._gradients
        )

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        train_op = super().apply_gradients(grads_and_vars, name, **kwargs)
        with tf.control_dependencies([train_op]):
            with tf.control_dependencies(
                    [
                        self._optimizer.iterations.assign_add(
                            tf.cast(
                                tf.where(self.step % self._accum_steps == 0, 1, 0), tf.int64
                            ),
                            read_value=False,
                        )
                    ]
            ):
                return self.step.assign_add(1, read_value=False)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        accum_gradient = self.get_slot(var, "ga")
        if accum_gradient is not None and grad is not None:
            accum_gradient.assign_add(
                grad, use_locking=self._use_locking, read_value=False
            )

        return self._apply_grad(accum_gradient, var, apply_state)

    def _resource_apply_sparse(self, grad: types.TensorLike, var, indices, apply_state):
        accum_gradient = self.get_slot(var, "ga")
        if accum_gradient is not None and grad is not None:
            self._resource_scatter_add(accum_gradient, indices, grad)

        return self._apply_grad(accum_gradient, var, apply_state)

    def _apply_grad(self, accum_gradient, var, apply_state):
        grad = tf.where(
            self.step % self._accum_steps == 0,
            accum_gradient,
            tf.zeros_like(var),
        )
        if "apply_state" in self._optimizer._dense_apply_args:
            train_op = self._optimizer._resource_apply_dense(
                grad,
                var,
                apply_state=apply_state,
            )
        else:
            train_op = self._optimizer._resource_apply_dense(grad, var)
        reset_val = tf.where(
            grad == accum_gradient, tf.zeros_like(accum_gradient), accum_gradient
        )
        reset_op = accum_gradient.assign(
            reset_val,
            use_locking=self._use_locking,
            read_value=False,
        )

        return tf.group(train_op, reset_op)

    def reset(self):
        """Resets the accumulated gradients on the current replica."""
        assign_ops = []
        if not self._gradients:
            return assign_ops

        for gradient in self._gradients:
            if gradient is not None:
                assign_ops.append(
                    gradient.assign(
                        tf.zeros_like(gradient),
                        use_locking=self._use_locking,
                        read_value=False,
                    )
                )

        return tf.group(assign_ops)

    @property
    def inner_optimizer(self):
        """The optimizer that this LossScaleOptimizer is wrapping."""
        return self._optimizer

    @property
    def iterations(self):
        return self._optimizer.iterations

    @iterations.setter
    def iterations(self, variable):
        self._optimizer.iterations = variable

    @property
    def lr(self):
        return self._optimizer._get_hyper("learning_rate")

    @lr.setter
    def lr(self, lr):
        self._optimizer._set_hyper("learning_rate", lr)  #

    @property
    def learning_rate(self):
        return self._optimizer._get_hyper("learning_rate")

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._optimizer._set_hyper("learning_rate", learning_rate)

    def get_config(self):
        config = {
            "accum_steps": self._accum_steps,
            "optimizer": tf.keras.optimizers.serialize(self._optimizer),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop("optimizer"), custom_objects=custom_objects
        )
        return cls(optimizer, **config)

