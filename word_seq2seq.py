
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense
from keras.utils.vis_utils import plot_model
from keras.preprocessing.text import text_to_word_sequence
import numpy as np



'''
 Training setting
 epochs = 50, num_samples = 10000
 
'''
batch_size = 64  # Batch size for training.
epochs = 50  # Number of epochs to train for. Iterations count of the loop of the training
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'twitter_data.txt'


# Vectorize the data.
input_texts = [[]]
target_texts = [[]]
input_words = set()
target_words = set()
lines = open(data_path).read().split('\n')



'''
 Word tokenization for input and output sentences
 
'''
for line in lines[: min(num_samples, len(lines) - 1)]:
    # Get context and response separately
    input_text, target_text = line.split('\t')
    
    # We use "startofasentence" as the "start sequence" word
    # for the targets, and "endofasentence" as "end sequence" word.
    target_text = 'startofasentence' + target_text + ' endofasentence'
    
    # text to word sequence
    input_text = text_to_word_sequence(input_text)
    target_text = text_to_word_sequence(target_text)

    input_texts.append(input_text)
    target_texts.append(target_text)
    
    # Word tokenization
    for word in input_text:
        if word not in input_words:
            input_words.add(word)
    for word in target_text:
        if word not in target_words:
            target_words.add(word)



# Get the freq of each word
input_words = sorted(list(input_words))
target_words = sorted(list(target_words))



# Remove the first empty elt
input_texts = input_texts[1:]
target_texts = target_texts[1:]


num_encoder_tokens = len(input_words)
num_decoder_tokens = len(target_words)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])


# Print
print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)



# Create dict for words
input_token_index = dict(
    [(word, i) for i, word in enumerate(input_words)])
target_token_index = dict(
    [(word, i) for i, word in enumerate(target_words)])


# Get input of encoder and decoder by one-hot vector initializer
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
# Get target of decoder by one-hot vector
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')



'''
 Create one-hot format of the data for training
 
'''
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, word in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[word]] = 1.
    for t, word in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[word]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[word]] = 1.



# Encoder
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]



# Decoder
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# Plot the model
plot_model(model, to_file='keras_model-word.png', show_shapes=True)

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)


# After finish training, save the model
model.save('s2s_10000_50.h5')

# # Load the model
# model.load_weights('s2s_10000_50.h5')


'''
 Inference setup
 
'''
# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


# Reverse-lookup token index to decode sequences back to readable sequence
reverse_input_word_index = dict((i, word) for word, i in input_token_index.items())
reverse_target_word_index = dict((i, word) for word, i in target_token_index.items())


'''
 Inference loop:
    1) Encode the input sentence and retrieve the initial decoder state
    2) Run one step of the decoder with this initial state and a "start of sequence" token as target. 
       The output will be the next target word.
    3) Append the target word predicted and repeat.
 
'''
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start word.
    target_seq[0, 0, target_token_index['startofasentence']] = 1.
        # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index[sampled_token_index]
        decoded_sentence += [sampled_word]

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_word == 'endofasentence' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


# Test the model: give input sentence, get output
for seq_index in range(10):
    # Take one sequence (part of the training test)
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('#')
    print('Input sentence:', ' '.join(input_texts[seq_index]))
    print('Decoded sentence:', ' '.join(decoded_sentence[:-1]))

