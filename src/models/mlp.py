import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Flatten, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError

def build_mlp_model(max_sequence_length, num_numeric_features, num_companies, num_targets_out, config):
    """
    Build MLP Model.
    
    Args:
        max_sequence_length (int): Input sequence length.
        num_numeric_features (int): Number of features per time step.
        num_companies (int): Total companies for embedding.
        num_targets_out (int): Output size.
        config (dict): Configuration dictionary (models.mlp section).
        
    Returns:
        tf.keras.Model: Compiled model.
    """
    # Params
    emb_dim = config['models']['common']['embedding_dim']
    dropout_rate = config['models']['common']['dropout_rate']
    lr = config['models']['common']['learning_rate']
    hidden_units = config['models']['mlp']['hidden_layers']
    
    # Inputs
    sequence_input = Input(shape=(max_sequence_length, num_numeric_features), name='Sequence_Input')
    company_input = Input(shape=(1,), name='Company_Input')
    
    # Process
    flat_seq = Flatten(name='Flatten_Sequence')(sequence_input)
    
    # Embedding
    company_emb = Embedding(input_dim=num_companies, output_dim=emb_dim, 
                            embeddings_initializer='he_normal', name='Company_Embedding')(company_input)
    flat_emb = Flatten(name='Flatten_Embedding')(company_emb)
    
    # Merge
    merged = Concatenate(name='Concatenate')([flat_seq, flat_emb])
    
    # MLP Layers
    x = merged
    for i, units in enumerate(hidden_units):
        x = Dense(units, activation='relu', kernel_initializer='he_normal', name=f'MLP_Hidden_{i+1}')(x)
        x = Dropout(dropout_rate)(x)
        
    # Output
    output = Dense(num_targets_out, name='Output_Layer')(x)
    
    model = Model(inputs=[sequence_input, company_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mse',
                  metrics=[RootMeanSquaredError(name='rmse')])
    
    return model
