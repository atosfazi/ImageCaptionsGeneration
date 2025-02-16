from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Add
from tensorflow.keras.models import Model


def build_model(max_seq_len):
    img_input = Input(shape=(2048,))
    img_dense = Dense(256, activation='relu')(img_input)

    text_input = Input(shape=(max_seq_len,))
    embedding = Embedding(input_dim=5000, output_dim=256, input_length=max_seq_len, mask_zero=True)(text_input)
    lstm = LSTM(256, return_sequences=False)(embedding)

    merged = Add()([img_dense, lstm])
    merged_dense = Dense(256, activation='relu')(merged)
    output = Dense(max_seq_len, activation='softmax')(merged_dense)

    model = Model(inputs=[img_input, text_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
