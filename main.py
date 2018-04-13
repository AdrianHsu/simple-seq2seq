import tensorflow as tf
import numpy as np
import helpers

tf.reset_default_graph()
sess = tf.InteractiveSession()

PAD = 0
EOS = 1

vocab_size = 10

n_input = 20 # already embedded
n_hidden_en = 20
n_hidden_de = n_hidden_en

max_time = 10
batch_size = 4

encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32)
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32)
decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32)

embeddings = tf.Variable(tf.random_uniform([vocab_size, n_input], -1.0, 1.0), dtype=tf.float32)

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

with tf.variable_scope('encoder'):
    encoder_cell = tf.contrib.rnn.LSTMCell(n_hidden_en)

    _, final_state_en = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded, dtype=tf.float32, time_major=True)

print(final_state_en)
with tf.variable_scope('decoder'):
    decoder_cell = tf.contrib.rnn.LSTMCell(n_hidden_de)

    outputs_de, final_state_de = tf.nn.dynamic_rnn(decoder_cell, decoder_inputs_embedded, initial_state=final_state_en, dtype=tf.float32, time_major=True)

decoder_logits = tf.contrib.layers.linear(outputs_de, vocab_size)

decoder_prediction = tf.argmax(decoder_logits, 2)

print(decoder_logits)

stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32), logits=decoder_logits)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())

# test 1
# batch_ = [[6], [3, 4], [9,8,7]]

# batch_, batch_length_ = helpers.batch(batch_)
# print('batch_encoded:\n' + str(batch_))

# din_, dlen_ = helpers.batch(np.ones(shape=(3, 1), dtype=np.int32),
#                             max_sequence_length=4)
# print('decoder inputs:\n' + str(din_))

# pred_ = sess.run(decoder_prediction,
#     feed_dict={
#         encoder_inputs: batch_,
#         decoder_inputs: din_,
#     })
# print('decoder predictions:\n' + str(pred_))


# test 2
batch_size = 100

batches = helpers.random_sequences(length_from=3, length_to=8,
   vocab_lower=2, vocab_upper=10,
   batch_size=batch_size)

print('head of the batch:')
for seq in next(batches)[:10]:
    print(seq)


def next_feed():
    batch = next(batches)
    encoder_inputs_, _ = helpers.batch(batch)
    decoder_targets_, _ = helpers.batch(
        [(sequence) + [EOS] for sequence in batch]
    )
    decoder_inputs_, _ = helpers.batch(
        [[EOS] + (sequence) for sequence in batch]
    )
    return {
        encoder_inputs: encoder_inputs_,
        decoder_inputs: decoder_inputs_,
        decoder_targets: decoder_targets_,
    }

loss_track = []

max_batches = 3001
batches_in_epoch = 1000

try:
    for batch in range(max_batches):
        fd = next_feed()
        _, l = sess.run([train_op, loss], fd)
        loss_track.append(l)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            predict_ = sess.run(decoder_prediction, fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(inp))
                print('    predicted > {}'.format(pred))
                if i >= 2:
                    break
            print()
except KeyboardInterrupt:
    print('training interrupted')


import matplotlib.pyplot as plt
plt.plot(loss_track)
print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))