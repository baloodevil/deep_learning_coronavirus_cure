import tensorflow as tf

import numpy as np
import os
import time


# Change the following line to run this code on your own data.
#path_to_file = tf.keras.utils.get_file('AllSentences.txt')

tokens = open('AllSentences.txt', 'rb').read().decode(encoding='utf-8').split()
# length of text is the number of characters in it
print ('Length of text: {} tokens'.format(len(tokens)))


# %%
# Take a look at the first 250 characters in text
print(tokens[:250])


# %%
# The unique tokens in the file
#vocab = sorted(set(text), reverse=True)
# set makes it a unique list
# list makes it an array
vocab = list(set(tokens))
print ('{} unique tokens'.format(len(vocab)))








# %% [markdown]
# ## Process the text
# %% [markdown]
# ### Vectorize the text

# Before training, we need to map strings to a numerical representation. Create two lookup tables: one mapping characters to numbers, and another for numbers to characters.

# %%
# Creating a mapping from unique characters to indices
token2idx = {u:get_vec(len(vocab),i) for i, u in enumerate(vocab)}
idx2token = np.array(vocab)



# text_as_int = np.array([token2idx[c] for c in tokens])

# # %% [markdown]
# # Now we have an integer representation for each character. Notice that we mapped the character as indexes from 0 to `len(unique)`.

# # %%
# print('{')
# for char,_ in zip(token2idx, range(20)):
#     print('  {:4s}: {:3d},'.format(repr(char), token2idx[char]))
# print('  ...\n}')


def get_vec(len_doc,word):
    empty_vector = [0] * len_doc
    vect = 0
    find = np.where( np.array(vocab) == word)[0][0]
    empty_vector[find] = 1
    return empty_vector

def get_matrix(vocab):
    mat = []
    len_doc = len(vocab)
    for i in tokens:
        vec = get_vec(len_doc,i)
        mat.append(vec)
        
    return np.asarray(mat)


matrix_data = get_matrix(vocab)

print ("\nMATRIX:")
print (matrix_data)








# # %%
# # Show how the first 13 characters from the text are mapped to integers
# print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

# %% [markdown]
# ### The prediction task
# %% [markdown]
# Given a character, or a sequence of characters, what is the most probable next character? This is the task we're training the model to perform. The input to the model will be a sequence of characters, and we train the model to predict the output—the following character at each time step.
# 
# Since RNNs maintain an internal state that depends on the previously seen elements, given all the characters computed until this moment, what is the next character?
# 
# %% [markdown]
# ### Create training examples and targets
# 
# Next divide the text into example sequences. Each input sequence will contain `seq_length` characters from the text.
# 
# For each input sequence, the corresponding targets contain the same length of text, except shifted one character to the right.
# 
# So break the text into chunks of `seq_length+1`. For example, say `seq_length` is 4 and our text is "Hello". The input sequence would be "Hell", and the target sequence "ello".
# 
# To do this first use the `tf.data.Dataset.from_tensor_slices` function to convert the text vector into a stream of character indices.

# %%
# The maximum length sentence we want for a single input in characters
seq_length = 4    # 100
#examples_per_epoch = len(tokens)//(seq_length+1)    # 24/11
examples_per_epoch = 24

# Create training examples / targets
word_dataset = tf.data.Dataset.from_tensor_slices(matrix_data) #text_as_int)

print('here')
for i in word_dataset.take(5):
  print(i)
  # print(idx2token[i.numpy()])
print('there')

# %% [markdown]
# The `batch` method lets us easily convert these individual characters to sequences of the desired size.

# %%
sequences = word_dataset.batch(seq_length, drop_remainder=True)

for item in sequences.take(5):
  print(item)
  #print(repr(''.join(idx2token[item.numpy()])))

# %% [markdown]
# For each sequence, duplicate and shift it to form the input and target text by using the `map` method to apply a simple function to each batch:

# %%
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# %% [markdown]
# Print the first examples input and target values:

# %%
for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2token[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2token[target_example.numpy()])))















# # %% [markdown]
# # Each index of these vectors are processed as one time step. For the input at time step 0, the model receives the index for "F" and trys to predict the index for "i" as the next character. At the next timestep, it does the same thing but the `RNN` considers the previous step context in addition to the current input character.

# # %%
# for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
#     print("Step {:4d}".format(i))
#     print("  input: {} ({:s})".format(input_idx, repr(idx2token[input_idx])))
#     print("  expected output: {} ({:s})".format(target_idx, repr(idx2token[target_idx])))

# # %% [markdown]
# # ### Create training batches
# # 
# # We used `tf.data` to split the text into manageable sequences. But before feeding this data into the model, we need to shuffle the data and pack it into batches.

# # %%
# # Batch size
# BATCH_SIZE = 64

# # Buffer size to shuffle the dataset
# # (TF data is designed to work with possibly infinite sequences,
# # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# # it maintains a buffer in which it shuffles elements).
# BUFFER_SIZE = 10000

# dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# dataset

# # %% [markdown]
# # ## Build The Model
# # %% [markdown]
# # Use `tf.keras.Sequential` to define the model. For this simple example three layers are used to define our model:
# # 
# # * `tf.keras.layers.Embedding`: The input layer. A trainable lookup table that will map the numbers of each character to a vector with `embedding_dim` dimensions;
# # * `tf.keras.layers.GRU`: A type of RNN with size `units=rnn_units` (You can also use a LSTM layer here.)
# # * `tf.keras.layers.Dense`: The output layer, with `vocab_size` outputs.

# # %%
# # Length of the vocabulary in chars
# vocab_size = len(vocab)

# # The embedding dimension
# embedding_dim = 256

# # Number of RNN units
# rnn_units = 1024


# # %%
# def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
#   model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(vocab_size, embedding_dim,
#                               batch_input_shape=[batch_size, None]),
#     tf.keras.layers.GRU(rnn_units,
#                         return_sequences=True,
#                         stateful=True,
#                         recurrent_initializer='glorot_uniform'),
#     tf.keras.layers.Dense(vocab_size)
#   ])
#   return model


# # %%
# model = build_model(
#   vocab_size = len(vocab),
#   embedding_dim=embedding_dim,
#   rnn_units=rnn_units,
#   batch_size=BATCH_SIZE)

# # %% [markdown]
# # For each character the model looks up the embedding, runs the GRU one timestep with the embedding as input, and applies the dense layer to generate logits predicting the log-likelihood of the next character:
# # 
# # ![A drawing of the data passing through the model](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/images/text_generation_training.png?raw=1)
# # %% [markdown]
# # ## Try the model
# # 
# # Now run the model to see that it behaves as expected.
# # 
# # First check the shape of the output:

# # %%
# for input_example_batch, target_example_batch in dataset.take(1):
#   example_batch_predictions = model(input_example_batch)
#   print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

# # %% [markdown]
# # In the above example the sequence length of the input is `100` but the model can be run on inputs of any length:

# # %%
# model.summary()

# # %% [markdown]
# # To get actual predictions from the model we need to sample from the output distribution, to get actual character indices. This distribution is defined by the logits over the character vocabulary.
# # 
# # Note: It is important to _sample_ from this distribution as taking the _argmax_ of the distribution can easily get the model stuck in a loop.
# # 
# # Try it for the first example in the batch:

# # %%
# sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
# sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

# # %% [markdown]
# # This gives us, at each timestep, a prediction of the next character index:

# # %%
# sampled_indices

# # %% [markdown]
# # Decode these to see the text predicted by this untrained model:

# # %%
# print("Input: \n", repr("".join(idx2token[input_example_batch[0]])))
# print()
# print("Next Char Predictions: \n", repr("".join(idx2token[sampled_indices ])))

# # %% [markdown]
# # ## Train the model
# # %% [markdown]
# # At this point the problem can be treated as a standard classification problem. Given the previous RNN state, and the input this time step, predict the class of the next character.
# # %% [markdown]
# # ### Attach an optimizer, and a loss function
# # %% [markdown]
# # The standard `tf.keras.losses.sparse_categorical_crossentropy` loss function works in this case because it is applied across the last dimension of the predictions.
# # 
# # Because our model returns logits, we need to set the `from_logits` flag.
# # 

# # %%
# def loss(labels, logits):
#   return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# example_batch_loss  = loss(target_example_batch, example_batch_predictions)
# print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
# print("scalar_loss:      ", example_batch_loss.numpy().mean())

# # %% [markdown]
# # Configure the training procedure using the `tf.keras.Model.compile` method. We'll use `tf.keras.optimizers.Adam` with default arguments and the loss function.

# # %%
# model.compile(optimizer='adam', loss=loss)

# # %% [markdown]
# # ### Configure checkpoints
# # %% [markdown]
# # Use a `tf.keras.callbacks.ModelCheckpoint` to ensure that checkpoints are saved during training:

# # %%
# # Directory where the checkpoints will be saved
# checkpoint_dir = './training_checkpoints'
# # Name of the checkpoint files
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_prefix,
#     save_weights_only=True)

# # %% [markdown]
# # ### Execute the training
# # %% [markdown]
# # To keep training time reasonable, use 10 epochs to train the model. In Colab, set the runtime to GPU for faster training.

# # %%
# EPOCHS=10


# # %%
# history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# # %% [markdown]
# # ## Generate text
# # %% [markdown]
# # ### Restore the latest checkpoint
# # %% [markdown]
# # To keep this prediction step simple, use a batch size of 1.
# # 
# # Because of the way the RNN state is passed from timestep to timestep, the model only accepts a fixed batch size once built.
# # 
# # To run the model with a different `batch_size`, we need to rebuild the model and restore the weights from the checkpoint.
# # 

# # %%
# tf.train.latest_checkpoint(checkpoint_dir)


# # %%
# model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

# model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

# model.build(tf.TensorShape([1, None]))


# # %%
# model.summary()

# # %% [markdown]
# # ### The prediction loop
# # 
# # The following code block generates the text:
# # 
# # * It Starts by choosing a start string, initializing the RNN state and setting the number of characters to generate.
# # 
# # * Get the prediction distribution of the next character using the start string and the RNN state.
# # 
# # * Then, use a categorical distribution to calculate the index of the predicted character. Use this predicted character as our next input to the model.
# # 
# # * The RNN state returned by the model is fed back into the model so that it now has more context, instead than only one character. After predicting the next character, the modified RNN states are again fed back into the model, which is how it learns as it gets more context from the previously predicted characters.
# # 
# # 
# # ![To generate text the model's output is fed back to the input](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/images/text_generation_sampling.png?raw=1)
# # 
# # Looking at the generated text, you'll see the model knows when to capitalize, make paragraphs and imitates a Shakespeare-like writing vocabulary. With the small number of training epochs, it has not yet learned to form coherent sentences.

# # %%
# def generate_text(model, start_string):
#   # Evaluation step (generating text using the learned model)

#   # Number of characters to generate
#   num_generate = 1000

#   # Converting our start string to numbers (vectorizing)
#   input_eval = [token2idx[s] for s in start_string]
#   input_eval = tf.expand_dims(input_eval, 0)

#   # Empty string to store our results
#   text_generated = []

#   # Low temperatures results in more predictable text.
#   # Higher temperatures results in more surprising text.
#   # Experiment to find the best setting.
#   temperature = 1.0

#   # Here batch size == 1
#   model.reset_states()
#   for i in range(num_generate):
#       predictions = model(input_eval)
#       # remove the batch dimension
#       predictions = tf.squeeze(predictions, 0)

#       # using a categorical distribution to predict the character returned by the model
#       predictions = predictions / temperature
#       predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

#       # We pass the predicted character as the next input to the model
#       # along with the previous hidden state
#       input_eval = tf.expand_dims([predicted_id], 0)

#       text_generated.append(idx2token[predicted_id])

#   return (start_string + ''.join(text_generated))


# # %%
# print(generate_text(model, start_string=u"ROMEO: "))

# # %% [markdown]
# # The easiest thing you can do to improve the results it to train it for longer (try `EPOCHS=30`).
# # 
# # You can also experiment with a different start string, or try adding another RNN layer to improve the model's accuracy, or adjusting the temperature parameter to generate more or less random predictions.
# # %% [markdown]
# # ## Advanced: Customized Training
# # 
# # The above training procedure is simple, but does not give you much control.
# # 
# # So now that you've seen how to run the model manually let's unpack the training loop, and implement it ourselves. This gives a starting point if, for example, to implement _curriculum learning_ to help stabilize the model's open-loop output.
# # 
# # We will use `tf.GradientTape` to track the gradients. You can learn more about this approach by reading the [eager execution guide](https://www.tensorflow.org/guide/eager).
# # 
# # The procedure works as follows:
# # 
# # * First, initialize the RNN state. We do this by calling the `tf.keras.Model.reset_states` method.
# # 
# # * Next, iterate over the dataset (batch by batch) and calculate the *predictions* associated with each.
# # 
# # * Open a `tf.GradientTape`, and calculate the predictions and loss in that context.
# # 
# # * Calculate the gradients of the loss with respect to the model variables using the `tf.GradientTape.grads` method.
# # 
# # * Finally, take a step downwards by using the optimizer's `tf.train.Optimizer.apply_gradients` method.
# # 
# # 

# # %%
# model = build_model(
#   vocab_size = len(vocab),
#   embedding_dim=embedding_dim,
#   rnn_units=rnn_units,
#   batch_size=BATCH_SIZE)


# # %%
# optimizer = tf.keras.optimizers.Adam()


# # %%
# @tf.function
# def train_step(inp, target):
#   with tf.GradientTape() as tape:
#     predictions = model(inp)
#     loss = tf.reduce_mean(
#         tf.keras.losses.sparse_categorical_crossentropy(
#             target, predictions, from_logits=True))
#   grads = tape.gradient(loss, model.trainable_variables)
#   optimizer.apply_gradients(zip(grads, model.trainable_variables))

#   return loss


# # %%
# # Training step
# EPOCHS = 10

# for epoch in range(EPOCHS):
#   start = time.time()

#   # initializing the hidden state at the start of every epoch
#   # initally hidden is None
#   hidden = model.reset_states()

#   for (batch_n, (inp, target)) in enumerate(dataset):
#     loss = train_step(inp, target)

#     if batch_n % 100 == 0:
#       template = 'Epoch {} Batch {} Loss {}'
#       print(template.format(epoch+1, batch_n, loss))

#   # saving (checkpoint) the model every 5 epochs
#   if (epoch + 1) % 5 == 0:
#     model.save_weights(checkpoint_prefix.format(epoch=epoch))

#   print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
#   print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

# model.save_weights(checkpoint_prefix.format(epoch=epoch))

