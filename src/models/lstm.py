import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Flatten, Dropout, LSTM, Masking
from tensorflow.keras.metrics import RootMeanSquaredError

def build_lstm_model(max_sequence_length, num_numeric_features, num_companies, num_targets_out, config):
    """
    Build LSTM Model.
    """
    # Params
    emb_dim = config['models']['common']['embedding_dim']
    dropout_rate = config['models']['common']['dropout_rate']
    lr = config['models']['common']['learning_rate']
    
    l1_units = config['models']['lstm']['layer_1_units']
    l2_units = config['models']['lstm']['layer_2_units']
    dense_units = config['models']['lstm']['dense_units']
    
    # Inputs
    sequence_input = Input(shape=(max_sequence_length, num_numeric_features), name='Sequence_Input')
    company_input = Input(shape=(1,), name='Company_Input')
    
    # Embedding
    company_emb = Embedding(input_dim=num_companies, output_dim=emb_dim, 
                            embeddings_initializer='he_normal', name='Company_Embedding')(company_input)
    flat_emb = Flatten(name='Flatten_Embedding')(company_emb)
    
    # LSTM Branch
    masked_input = Masking(mask_value=0.0)(sequence_input)
    
    x = LSTM(l1_units, kernel_initializer='he_normal', return_sequences=True, name='LSTM_1')(masked_input)
    x = Dropout(dropout_rate)(x)
    
    x = LSTM(l2_units, kernel_initializer='he_normal', return_sequences=False, name='LSTM_2')(x)
    x = Dropout(dropout_rate)(x)
    
    # Merge with Embedding
    merged = Concatenate(name='Merge')([x, flat_emb])
    
    # Dense
    x = Dense(dense_units, activation='relu', kernel_initializer='he_normal', name='Dense_Post_Merge')(merged)
    x = Dropout(dropout_rate)(x)
    
    # Output
    output = Dense(num_targets_out, name='Output_Layer')(x)
    
    model = Model(inputs=[sequence_input, company_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mse',
                  metrics=[RootMeanSquaredError(name='rmse')])
    
    return model
