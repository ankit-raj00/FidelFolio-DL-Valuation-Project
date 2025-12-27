import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Flatten, Dropout, LSTM, Masking
from tensorflow.keras.metrics import RootMeanSquaredError

def build_encoder_decoder_model(max_sequence_length, num_numeric_features, num_companies, num_targets_out, config):
    """
    Build Encoder-Decoder Model.
    """
    # Params
    emb_dim = config['models']['common']['embedding_dim']
    dropout_rate = config['models']['common']['dropout_rate']
    lr = config['models']['common']['learning_rate']
    
    enc_units = config['models']['encoder_decoder']['encoder_units']
    dense_units = config['models']['encoder_decoder']['dense_units']
    
    # Inputs
    sequence_input = Input(shape=(max_sequence_length, num_numeric_features), name='Sequence_Input')
    company_input = Input(shape=(1,), name='Company_Input')
    
    # Embedding
    company_emb = Embedding(input_dim=num_companies, output_dim=emb_dim, 
                            embeddings_initializer='he_normal', name='Company_Embedding')(company_input)
    flat_emb = Flatten(name='Flatten_Embedding')(company_emb)
    
    # Encoder (LSTM)
    masked_input = Masking(mask_value=0.0)(sequence_input)
    # We only need the final state (context)
    _, state_h, state_c = LSTM(enc_units, kernel_initializer='he_normal', 
                               return_state=True, name='Encoder_LSTM')(masked_input)
                               
    encoder_context = state_h # Using hidden state as context
    
    # Decoder (simplified to Dense for regression)
    # Combine Context + Company Info
    merged = Concatenate(name='Context_Merge')([encoder_context, flat_emb])
    
    x = Dense(dense_units, activation='relu', kernel_initializer='he_normal', name='Decoder_Dense')(merged)
    x = Dropout(dropout_rate)(x)
    
    # Output
    output = Dense(num_targets_out, name='Output_Layer')(x)
    
    model = Model(inputs=[sequence_input, company_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mse',
                  metrics=[RootMeanSquaredError(name='rmse')])
    
    return model
