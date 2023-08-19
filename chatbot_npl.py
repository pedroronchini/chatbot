import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# Carregando o JSON e processando as informações do json
with open('dev-v1.1.json', 'r') as json_file:
    data = json.load(json_file)
    paragraphs = [p['context'] for entry in data['data'] for p in entry['paragraphs']]
    questions = [q['question'] for entry in data['data'] for p in entry['paragraphs'] for q in p['qas']]
    answers = [a['answers'][0]['text'] for entry in data['data'] for p in entry['paragraphs'] for a in p['qas']]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(paragraphs + questions + answers)
vocab_size = len(tokenizer.word_index) + 1

max_seq_length = 1

input_paragraphs = tokenizer.texts_to_sequences(paragraphs)
input_paragraphs = pad_sequences(input_paragraphs, maxlen=max_seq_length, padding='post')

input_questions = tokenizer.texts_to_sequences(questions)
input_questions = pad_sequences(input_questions, maxlen=max_seq_length, padding='post')

output_answers = tokenizer.texts_to_sequences(answers)
output_answers_input = pad_sequences(output_answers, maxlen=max_seq_length, padding='post', truncating='post')
output_answers_output = [text[1:] for text in output_answers_input]  # Shift the target output by 1
output_answers_output = pad_sequences(output_answers_output, maxlen=max_seq_length, padding='post', truncating='post')
output_answers_output = to_categorical(output_answers_output, num_classes=vocab_size)

# Criar o modelo de chatbot
input_layer = Input(shape=(max_seq_length,))
embedding_layer = Embedding(vocab_size, 128)(input_layer)
encoder_lstm = LSTM(128)(embedding_layer)
decoder_lstm = LSTM(128, return_sequences=True)(embedding_layer, initial_state=[encoder_lstm, encoder_lstm])
output_layer = Dense(vocab_size, activation='softmax')(decoder_lstm)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(np.array(input_paragraphs), np.array(output_answers_output), epochs=100, batch_size=1)

# Fazer previsões
def generate_response(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_seq_length, padding='post')
    predicted_id = np.argmax(model.predict(input_seq), axis=-1)
    return tokenizer.sequences_to_texts(predicted_id)[0]

# Teste
user_input = "Which NFL team represented the AFC at Super Bowl 50?"
response = generate_response(user_input)
print("Bot:", response)