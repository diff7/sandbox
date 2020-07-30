import time
import tensorflow as tf
import matplotlib.pyplot as plt

from transformer import Transformer, CustomSchedule

params = {
    "num_layers": 4,
    "d_model": 128,
    "dff": 512,
    "num_heads": 8,
    "input_vocab_size": 0,
    "target_vocab_size": 0,
    "dropout_rate": 0.1,
    "checkpoint_path": "./checkpoints/train",
}


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


# triangular matrix
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


class Trainer:
    def __init__(self, params):
        self.params = params
        self.model = Transformer(
            params["num_layers"],
            params["d_model"],
            params["num_heads"],
            params["dff"],
            params["input_vocab_size"],
            params["target_vocab_size"],
            pe_input=params["input_vocab_size"],
            pe_target=params["target_vocab_size"],
            rate=params["dropout_rate"],
        )

        learning_rate = CustomSchedule(params["d_model"])
        self.opt = tf.keras.optimizers.Adam(
            learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )

        ckpt = tf.train.Checkpoint(transformer=self.model, optimizer=self.opt)

        self.ckpt_manager = tf.train.CheckpointManager(
            ckpt, params["checkpoint_path"], max_to_keep=5
        )
        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Latest checkpoint restored!!")

    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.

    @tf.function(input_signature=train_step_signature)
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self.model(
                inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask
            )
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)

    def train_epochs(self, EPOCHS, train_dataset):
        for epoch in range(EPOCHS):
            start = time.time()

            train_loss.reset_states()
            train_accuracy.reset_states()

            # inp -> portuguese, tar -> english
            for (batch, (inp, tar)) in enumerate(train_dataset):
                self.train_step(inp, tar)

                if batch % 50 == 0:
                    print(
                        "Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
                            epoch + 1,
                            batch,
                            train_loss.result(),
                            train_accuracy.result(),
                        )
                    )

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print(
                    "Saving checkpoint for epoch {} at {}".format(
                        epoch + 1, ckpt_save_path
                    )
                )

            print(
                "Epoch {} Loss {:.4f} Accuracy {:.4f}".format(
                    epoch + 1, train_loss.result(), train_accuracy.result()
                )
            )

            print("Time taken for 1 epoch: {} secs\n".format(time.time() - start))

    def evaluate(self, inp_sentence, tokenizer_inp, tokenizer_tar):
        start_token = [tokenizer_inp.vocab_size]
        end_token = [tokenizer_inp.vocab_size + 1]

        # inp sentence is portuguese, hence adding the start and end token
        inp_sentence = start_token + tokenizer_inp.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoder_input = [tokenizer_tar.vocab_size]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(self.params["MAX_LENGTH"]):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output
            )

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.model(
                encoder_input,
                output,
                False,
                enc_padding_mask,
                combined_mask,
                dec_padding_mask,
            )

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == tokenizer_tar.vocab_size + 1:
                return tf.squeeze(output, axis=0), attention_weights

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def plot_attention_weights(
        self, attention, sentence, result, layer, tokenizer_inp, tokenizer_tar
    ):

        fig = plt.figure(figsize=(16, 8))

        sentence = tokenizer_inp.encode(sentence)

        attention = tf.squeeze(attention[layer], axis=0)

        for head in range(attention.shape[0]):
            ax = fig.add_subplot(2, 4, head + 1)

            # plot the attention weights
            ax.matshow(attention[head][:-1, :], cmap="viridis")

            fontdict = {"fontsize": 10}

            ax.set_xticks(range(len(sentence) + 2))
            ax.set_yticks(range(len(result)))

            ax.set_ylim(len(result) - 1.5, -0.5)

            ax.set_xticklabels(
                ["<start>"] + [tokenizer_inp.decode([i]) for i in sentence] + ["<end>"],
                fontdict=fontdict,
                rotation=90,
            )

            ax.set_yticklabels(
                [
                    tokenizer_tar.decode([i])
                    for i in result
                    if i < tokenizer_tar.vocab_size
                ],
                fontdict=fontdict,
            )

            ax.set_xlabel("Head {}".format(head + 1))

        plt.tight_layout()
        plt.show()

    def translate(self, sentence, tokenizer_tar, tokenizer_inp, plot=""):
        # e.g plot = 'decoder_layer4_block2'
        result, attention_weights = self.evaluate(
            sentence, tokenizer_inp, tokenizer_tar
        )

        predicted_sentence = tokenizer_tar.decode(
            [i for i in result if i < tokenizer_tar.vocab_size]
        )

        print("Input: {}".format(sentence))
        print("Predicted translation: {}".format(predicted_sentence))

        if plot:
            self.plot_attention_weights(
                attention_weights, sentence, result, plot, tokenizer_inp, tokenizer_tar
            )
