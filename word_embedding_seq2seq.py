from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Embedding, Dropout
from keras.utils.vis_utils import plot_model
import data
import numpy as np

# Load the dictionaries, contexts and responses
metadata, idx_q, idx_a, idx_target_decoder = data.load_data()

# Number of encoder/decoder tokens
num_encoder_tokens = len(metadata['idx2w'])
num_decoder_tokens = num_encoder_tokens

# Hyperparameters
latent_dim = 256
batch_size = 32
epochs = 20
dropout_rate = 0.3

# Prepare the input and output for training
encoder_input_data = idx_q
decoder_input_data = idx_a
decoder_target_data = data.one_hot_encode(idx_target_decoder, num_decoder_tokens)

# ENCODER SETUP
# Define an input layer, convert the input with the help of Embedding layer
# and connect to LSTM 
encoder_inputs = Input(shape=(None,))
x = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
x = Dropout(dropout_rate)(x)
x, state_h, state_c = LSTM(latent_dim, return_state=True)(x)
encoder_states = [state_h, state_c]

# DECODER SETUP
# Use 'encoder_states' as initial state.
decoder_inputs = Input(shape=(None,))
x = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
x = Dropout(dropout_rate)(x)
x = LSTM(latent_dim, return_sequences=True)(x, initial_state=encoder_states)
decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)

# BUILDING MODEL
# Turns 'encoder_input_data' & 'decoder_input_data' into 'decoder_target_data'
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# PLOT MODEL
plot_model(model, to_file='model-word-report.png', show_shapes=True)

# LOAD model weights for subsequent runs
# model.load_weights('chatbot.h5')

# Compile and run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

# SAVE MODEL for later use
model.save('chatbot.h5')

# INFERENCE ENCODER MODEL
encoder_model = Model(encoder_inputs, encoder_states)

# Inference decoder states and layers
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
x = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
x, state_h, state_c = LSTM(latent_dim, return_state=True)(x, initial_state=decoder_states_inputs)
decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)
decoder_states = [state_h, state_c]

# INFERENCE DECODER MODEL
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Plot the models
plot_model(encoder_model, to_file='encoder_inference_word.png', show_shapes=True)
plot_model(decoder_model, to_file='decoder_inference_word.png', show_shapes=True)

# SOME STATISTICS
# print('Number of samples:', encoder_input_data.shape[0])
# print('Number of unique input tokens:', num_encoder_tokens)
# print('Number of unique output tokens:', num_decoder_tokens)
# print('Max sequence length for inputs:', metadata['limit']['maxq'])
# print('Max sequence length for outputs:', metadata['limit']['maxa'])

# TEST SENTENCE
test_sentence = "yeah i'm preparing myself to drop a lot on this man, but definitely need something reliable"
test_sentence = data.filter_line(test_sentence.lower(), data.EN_WHITELIST)
onehot_test_sentence = data.zero_pad_and_hot_encode(test_sentence, metadata['w2idx'], num_decoder_tokens)

# Initial prediction for decoding
states_value = encoder_model.predict(onehot_test_sentence)

# Generate empty target sequence of length 1.
target_seq = np.zeros((1, num_decoder_tokens))
target_seq[0, metadata['w2idx']['<str>']] = 1.

stop_condition = False
decoded_sentence = []
while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a word
        sampled_token_index = np.argmax(output_tokens[0, :])
        sampled_token = metadata['idx2w'][sampled_token_index]
        decoded_sentence.append(sampled_token)

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == '<end>' or len(decoded_sentence) > 20):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, num_decoder_tokens))
        target_seq[0, sampled_token_index] = 1.        

        # Update states
        states_value = [h, c]

print(decoded_sentence)