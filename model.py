import tensorflow as tf
import keras_core as keras
from keras_nlp.layers import SinePositionEncoding
from keras.layers import *
from keras.activations import *

from utils_ops import *

class Transformer(keras.Model):
    """
    A Transformer model for sequence-to-sequence tasks.

    Args:
        vocab_size (int): The size of the vocabulary. Defaults to 256.
        emb_dim (int): The dimension of the embedding layer. Defaults to 512.
        num_stack (int): The number of stacked layers in the encoder and decoder. Defaults to 16.
        intermediate_dim (int): The dimension of the intermediate layer in the encoder and decoder. Defaults to 512.
        num_heads (int): The number of attention heads in the encoder and decoder. Defaults to 8.
        head_dim (int): The dimension of each attention head in the encoder and decoder. Defaults to 256.
        proj_dim (int): The dimension of the projection layer. If None, no projection layer is used. Defaults to None.
        dropout (float): The dropout rate. Defaults to 0.05.
        preprocess_fn (function): A function to preprocess the input. Defaults to lambda x : x.
        activation (function): The activation function. Defaults to relu.

    Attributes:
        vocab_size (int): The size of the vocabulary.
        emb_dim (int): The dimension of the embedding layer.
        num_stack (int): The number of stacked layers in the encoder and decoder.
        intermediate_dim (int): The dimension of the intermediate layer in the encoder and decoder.
        num_heads (int): The number of attention heads in the encoder and decoder.
        dropout (float): The dropout rate.
        activation (function): The activation function.
        epsilon (float): A small value to avoid division by zero.
        support_masking (bool): Whether the model supports masking.
        head_dim (int): The dimension of each attention head in the encoder and decoder.
        _layers (list): A list of layers in the model.
        embedding_layer (Embedding): The embedding layer.
        positional_encoder (function): The positional encoding function.
        encoder_layer (EncoderStacked): The encoder layer.
        decoder_layer (DecoderStacked): The decoder layer.
        projection_layer (Dense): The projection layer.
        proj_dim (int): The dimension of the projection layer.
        proj_activation (function): The activation function for the projection layer.

    Methods:
        build(encoder_input_shape, decoder_input_shape): Builds the model.
        call(encoder_input, decoder_input, encoder_padding_mask, decoder_padding_mask, encoder_attention_mask, decoder_attention_mask, training): Computes the output of the model.
    """
    
    def __init__(self, vocab_size=256, emb_dim=512, num_stack=64, intermediate_dim=512, num_heads=64, 
                 head_dim=512, proj_dim=None, dropout=0.05, preprocess_fn=lambda x : x, activation=relu, **kwargs):

        super(Transformer, self).__init__(**kwargs)

        self.vocab_size = vocab_size # default to 256 char-level
        self.emb_dim = emb_dim
        self.num_stack = num_stack 
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.epsilon = 1e-6
        self.support_masking = True
        self.head_dim = head_dim
        self._layers = []

        self.embedding_layer = None
        self.positional_encoder = None 
        self.encoder_layer = None 
        self.decoder_layer = None 
        self.projection_layer = None 
        self.proj_dim = proj_dim
        self.proj_activation = softmax

        self.preprocess_fn = preprocess_fn
        self.positional_encoding = lambda x: positional_encoding(x, wave_length=10000)

    def build(self, encoder_input_shape, decoder_input_shape):
        
        self.embedding_layer = Embedding(input_dim=self.vocab_size, 
                                         output_dim=self.emb_dim, 
                                         mask_zero=True, 
                                         name="embedding"
                                        )

        self.encoder_layer = EncoderStacked(num_stack=self.num_stack, 
                                      intermediate_dim=self.intermediate_dim,
                                      num_heads=self.num_heads,
                                      head_dim=self.head_dim,
                                      dropout=self.dropout,
                                      activation=self.activation,
                                      name="encoder"
                                      )
        self.decoder_layer = DecoderStacked(num_stack=self.num_stack,
                                      intermediate_dim=self.intermediate_dim,
                                      num_heads=self.num_heads,
                                      head_dim=self.head_dim,
                                      dropout=self.dropout,
                                      activation=self.activation,
                                      name="decoder"
                                      )
        if self.proj_dim is not None:

            self.projection_layer = Dense(self.proj_dim, name="output_projection")
            self.projection_layer.build(self.intermediate_dim)
        
        self.built = True

    def call(self, 
             encoder_input, 
             decoder_input, 
             encoder_padding_mask=None, 
             decoder_padding_mask=None, 
             encoder_attention_mask=None,
             decoder_attention_mask=None,
             training=True, 
             **kwargs):
        ''' 
        encoder_input: (batch, seq_len), decoder_input: (batch, seq_len), encoder_padding_mask: (batch, seq_len)
        decoder_padding_mask: (batch, seq_len)
        encoder_attention_mask: (batch, seq_len, seq_len) # if None the apply tensor of ones
        decoder_attention_mask: (batch, sseq_len, seq_len) # if None the apply tensor of ones
        '''

        encoder_seq_len = tf.shape(encoder_input)[-1]
        decoder_seq_len = tf.shape(decoder_input)[-1]
        batch = tf.shape(encoder_input)[0]
        #Preprocess
        encoder_input = self.preprocess_fn(encoder_input)
        decoder_input = self.preprocess_fn(decoder_input)
        #Embedding
        encoder_input = self.embedding_layer(encoder_input)
        decoder_input = self.embedding_layer(decoder_input)

        #Default masking to ones
        if encoder_padding_mask is None:
            if hasattr(encoder_input, "_keras_mask"):
                encoder_padding_mask = tf.cast(encoder_input._keras_mask, tf.int32)
            else:
                encoder_padding_mask = tf.ones(encoder_seq_len, dtype=tf.int32)
        
        if decoder_padding_mask is None:
            if hasattr(encoder_input, "_keras_mask"):
                decoder_padding_mask = tf.cast(decoder_input._keras_mask, tf.int32)
            else:
                decoder_padding_mask = tf.ones(decoder_seq_len, dtype=tf.int32)
        
        if encoder_attention_mask is None:
            encoder_attention_mask = tf.ones((batch, encoder_seq_len, encoder_seq_len), dtype=tf.int32)

        if decoder_attention_mask is None:
            decoder_attention_mask = tf.ones((batch, decoder_seq_len, decoder_seq_len), dtype=tf.int32)
        
        #Positional Encoding
        encoder_input = self.positional_encoding(encoder_input)
        decoder_input = self.positional_encoding(decoder_input)
        #Encoder Layer
        encoder_input = self.encoder_layer(encoder_input, 
                                   padding_mask=encoder_padding_mask, 
                                   attention_mask=encoder_attention_mask, 
                                   training=training
                                   )
        #Decoder Layer
        x_dec = self.decoder_layer(decoder_input,
                                    encoder_input,
                                    decoder_padding_mask=decoder_padding_mask,
                                    decoder_attention_mask=decoder_attention_mask,
                                    encoder_padding_mask=encoder_padding_mask,
                                    encoder_attention_mask=encoder_attention_mask,
                                    training=training
                                    )
        #Projection Layer
        if self.projection_layer is not None:
            
            x_dec = self.projection_layer(x_dec)
            x_dec = self.proj_activation(x_dec)

        return x_dec      

    def count_params(self):
            
        n_params = self.embedding_layer.count_params()  + \
                self.encoder_layer.count_params() + self.decoder_layer.count_params()
        
        if self.projection_layer is not None:
            n_params += self.projection_layer.count_params()
        
        return n_params
    
    def get_config_layer(self):

        layer_config = None

        return layer_config 

class Encoder(keras.layers.Layer):

    def __init__(self, intermediate_dim=512, num_heads=8, head_dim=256, dropout=0, activation=relu, **kwargs):

        super(Encoder, self).__init__(**kwargs)

        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.epsilon = 1e-6
        self.support_masking = True
        self.head_dim = head_dim

    def build(self, encoder_input_shape):

        hidden_dim = encoder_input_shape[-1]

        if self.head_dim is None: 
            self.head_dim = int(hidden_dim// self.num_heads)

        self._attention_layer = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.head_dim,
            value_dim=self.head_dim,
            dropout=self.dropout,
            output_shape=hidden_dim,
            name="attention_layer"
        )

        self._attention_norm = LayerNormalization(epsilon=self.epsilon, name="attention_norm")  
        self._attention_dropout = Dropout(self.dropout, name="attention_dropout")
        self._dense_norm = LayerNormalization(epsilon=self.epsilon, name="dense_norm")
        self._dense_dropout = Dropout(self.dropout, name="dense_dropout")
        self._dense_intermediate = Dense(self.intermediate_dim, activation=self.activation, name="dense_intermediate")
        self._dense_output = Dense(hidden_dim, activation=self.activation, name="dense_output")

        # Build layers
        self._attention_layer.build((encoder_input_shape, encoder_input_shape))
        self._attention_norm.build(encoder_input_shape)
        self._dense_intermediate.build(encoder_input_shape)

        intermediate_shape = list(encoder_input_shape)
        intermediate_shape[-1] = self.intermediate_dim
        self._dense_output.build(tuple(intermediate_shape))

        self.built = True

    def call(self, encoder_input, padding_mask=None, attention_mask=None, mask=None, training=True):
       
        self_attention_mask = add_padd_mask(padding_mask, attention_mask)

        residual = encoder_input
        attention_output, attention_score = self._attention_layer(query=encoder_input,
                                                                  value=encoder_input, 
                                                                  key=encoder_input,
                                                                  attention_mask=self_attention_mask,
                                                                  return_attention_scores=True,
                                                                  training=training, 
                                                                  use_causal_mask=False)
        x = attention_output 
        x = self._attention_dropout(x)
        x = self._attention_norm(x)
        x = x + residual

        residual = x
        x = self._dense_intermediate(x)
        x = self._dense_output(x)
        x = self._dense_dropout(x)
        x = x + residual
        x = self._dense_norm(x)

        return x, attention_score
    
    def compute_mask(self, inputs, mask=None):

        return mask
    
    def count_params(self):
        
        n_params = 0
        for attribute in self.__dict__.values():
            if hasattr(attribute, "trainable_variables"):
                n_params += sum([tf.size(w).numpy() for w in attribute.trainable_variables])
        
        return n_params
    
class EncoderStacked(keras.Model):
    
    def __init__(self, num_stack=8, intermediate_dim=512, num_heads=8, head_dim=256, dropout=0, activation=relu, **kwargs):

        super(EncoderStacked, self).__init__(**kwargs)

        self.num_stack = num_stack
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.epsilon = 1e-6
        self.support_masking = True
        self.head_dim = head_dim
        self._layers = []

        for i in range(self.num_stack):
            setattr(self, f"encoder_{i}", Encoder(
                intermediate_dim=self.intermediate_dim,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dropout=self.dropout,
                activation=self.activation,
                name=f"encoder_{i}"
            ))

    def count_params(self):
        n_params = sum([getattr(self, f"encoder_{i}").count_params() for i in range(self.num_stack)])
          
        return n_params

    def build(self, encoder_input_shape):
            
            for i in range(self.num_stack):
    
                getattr(self, f"encoder_{i}").build(encoder_input_shape)
                
            self.built = True
        
    def call(self, encoder_input, padding_mask=None, attention_mask=None, mask=None, training=True):

        x = encoder_input 

        for i in range(self.num_stack):

            x, _ = getattr(self, f"encoder_{i}")(x, 
                                                 padding_mask=padding_mask, 
                                                 attention_mask=attention_mask, 
                                                 mask=mask, 
                                                 training=training)
            
        return x

class Decoder(keras.layers.Layer): 

    def __init__(self, intermediate_dim=512, num_heads=8, head_dim=256, dropout=0, activation=relu, **kwargs):

        super(Decoder, self).__init__(**kwargs)

        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.epsilon = 1e-6
        self.support_masking = True
        self.head_dim = head_dim

    def build(self, decoder_input_shape, encoder_input_shape):

        hidden_dim = decoder_input_shape[-1]

        if self.head_dim is None: 
            self.head_dim = int(hidden_dim// self.num_heads)

        self._attention_layer = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.head_dim,
            value_dim=self.head_dim,
            dropout=self.dropout,
            output_shape=hidden_dim,
            name="attention_layer"
        )

        self._cross_attention_layer = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.head_dim,
            value_dim=self.head_dim,
            dropout=self.dropout,
            output_shape=hidden_dim,
            name="cross_attention_layer"
        )

        self._attention_norm = LayerNormalization(epsilon=self.epsilon, name="attention_norm")  
        self._cross_attention_norm = LayerNormalization(epsilon=self.epsilon, name="cross_attention_norm")
        self._dense_norm = LayerNormalization(epsilon=self.epsilon, name="dense_norm")
        self._attention_dropout = Dropout(self.dropout, name="attention_dropout")   
        self._cross_attention_dropout = Dropout(self.dropout, name="cross_attention_dropout")   
        self._dense_dropout = Dropout(self.dropout, name="dense_dropout")
        self._dense_intermediate = Dense(self.intermediate_dim, activation=self.activation, name="dense_intermediate")
        self._dense_output = Dense(hidden_dim, activation=self.activation, name="dense_output") 
        
        # Build layers
        self._attention_layer.build((decoder_input_shape, decoder_input_shape))
        self._attention_norm.build(decoder_input_shape)

        self._cross_attention_layer.build((encoder_input_shape, encoder_input_shape))
        self._cross_attention_norm.build(decoder_input_shape)

        self._dense_intermediate.build(decoder_input_shape)

        intermediate_shape = list(decoder_input_shape)
        intermediate_shape[-1] = self.intermediate_dim
        self._dense_output.build(tuple(intermediate_shape))
        self._dense_norm.build(decoder_input_shape)
        
        self.built = True

    def call(self, decoder_input,
                    encoder_input,
                    decoder_padding_mask=None,
                    decoder_attention_mask=None,
                    encoder_padding_mask=None,
                    encoder_attention_mask=None,
                    use_causal_mask=True, 
                    training=True,
                    mask=None):

        self_attention_mask = compute_attention_mask(decoder_input, 
                                                     decoder_padding_mask, 
                                                     decoder_attention_mask, 
                                                     use_causal_mask)

        x = decoder_input  # Intermediate result.
        # Self attention block.
        residual = x
        attention_output, self_attention_score = self._attention_layer(query=x, 
                                                value=x,
                                                key=x,
                                                attention_mask=self_attention_mask,
                                                return_attention_scores=True,
                                                training=training,
                                                use_causal_mask=True)
        
        x = attention_output 
        x = self._attention_dropout(x)
        x = self._attention_norm(x)
        x = x + residual

        # Compute cross attention mask.
        cross_attention_mask = add_padd_mask(encoder_padding_mask, encoder_attention_mask)
        # Cross attention block.
        residual = x

        attention_output, cross_attention_score = self._cross_attention_layer(query=x, 
                                                        value=encoder_input, 
                                                        key=encoder_input,
                                                        attention_mask=cross_attention_mask,
                                                        return_attention_scores=True,
                                                        training=training,
                                                        use_causal_mask=True)
        
        x = attention_output
        x = self._cross_attention_dropout(x)
        x = x + residual
        x = self._cross_attention_norm(x)

        residual = x

        x = self._dense_intermediate(x)
        x = self._dense_output(x)
        x = self._dense_dropout(x)
        x = x + residual
        x = self._dense_norm(x)

        return x, self_attention_score, cross_attention_score

    def compute_mask(self, inputs, mask=None):
        return mask

    def count_params(self):
        
        n_params = 0
        for attribute in self.__dict__.values():
            if hasattr(attribute, "trainable_variables"):
                n_params += sum([tf.size(w).numpy() for w in attribute.trainable_variables])
        
        return n_params

class DecoderStacked(keras.Model):

    def __init__(self, num_stack=8, intermediate_dim=512, num_heads=8, head_dim=256, dropout=0, activation=relu, **kwargs):
        super(DecoderStacked, self).__init__(**kwargs)

        self.num_stack = num_stack
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.epsilon = 1e-6
        self.support_masking = True
        self.head_dim = head_dim
        self._layers = []

        for i in range(self.num_stack):
            setattr(self, f"decoder_{i}", Decoder(
                intermediate_dim=self.intermediate_dim,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dropout=self.dropout,
                activation=self.activation,
                name=f"decoder_{i}"
            ))
        
    def build(self, decoder_input_shape, encoder_input_shape):

        for i in range(self.num_stack):

            getattr(self, f"decoder_{i}").build(decoder_input_shape, encoder_input_shape)
            
        self.built = True

    #TODO check input 
    def call(self, decoder_input, 
             encoder_input, 
             decoder_padding_mask=None, 
             decoder_attention_mask=None, 
             encoder_padding_mask=None, 
             encoder_attention_mask=None, 
             use_causal_mask=True, 
             training=True, 
             mask=None):

        for i in range(self.num_stack):

            decoder_input, _, _ = getattr(self, f"decoder_{i}")(decoder_input=decoder_input,
                                                                encoder_input=encoder_input, 
                                                                decoder_padding_mask=decoder_padding_mask,
                                                                decoder_attention_mask=decoder_attention_mask,
                                                                encoder_padding_mask=encoder_padding_mask,
                                                                encoder_attention_mask=encoder_attention_mask,
                                                                use_causal_mask=use_causal_mask,
                                                                training=training,
                                                                mask=mask)
                                                                
        return decoder_input
    
    def count_params(self):
        n_params = sum([getattr(self, f"decoder_{i}").count_params() for i in range(self.num_stack)])
          
        return n_params

    def compute_mask(self, inputs, mask=None):
        
        return mask

