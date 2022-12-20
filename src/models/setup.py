from .rnn import LSTMRegressor
from .transformer import LSTMTransformerRegressor

LSTM = 'lstm'
LSTM_TRANSFORMER = 'lstm_transformer'


def setup_model(params):
    if params.model_type == LSTM:
        return LSTMRegressor(params)
    elif params.model_type == LSTM_TRANSFORMER:
        return LSTMTransformerRegressor(params)
    else:
        raise NotImplementedError(f'Model type {params.model_type} is not implemented')