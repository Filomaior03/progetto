import numpy as np
import keras
from keras.layers import Input, Dense, LSTM, Conv1D, Flatten, Bidirectional, BatchNormalization, Activation, Multiply, Permute, add
from keras.models import Model
from keras.callbacks import EarlyStopping

from baselines import evaluate

#function to fit and evaluate models
def fit_and_evaluate_deep(model, train_X, train_y, valid_X, valid_y, test_X, test_y, target_min, target_max, epochs=100, batch_size=64, patience=15):
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

  early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

  history = model.fit(train_X, train_y, validation_data=(valid_X, valid_y), epochs=epochs, batch_size=batch_size, callbacks=[early_stop])

  valid_pred_norm = model.predict(valid_X).flatten()
  test_pred_norm = model.predict(test_X).flatten()

  valid_pred_mm = valid_pred_norm * (target_max - target_min) + target_min
  test_pred_mm = test_pred_norm * (target_max - target_min) + target_min

  valid_pred_mm = np.clip(valid_pred_mm, a_min=0, a_max=None)
  test_pred_mm = np.clip(test_pred_mm, a_min=0, a_max=None)

  valid_true_mm = valid_y * (target_max - target_min) + target_min
  test_true_mm = test_y * (target_max - target_min) + target_min

  valid_mae, valid_rmse, valid_r2 = evaluate(valid_true_mm, valid_pred_mm)
  test_mae, test_rmse, test_r2 = evaluate(test_true_mm, test_pred_mm)

  #MODIFICATO: aggiunti i 4 array denormalizzati (stesso pattern di fit_and_evaluate in
  #baselines.py) per poter ricalcolare le metriche per anno senza rifare il training.
  #ATTENZIONE: `history` è stato spostato in fondo alla tupla (era in posizione [6], ora è
  #in [10]) — se lo usi altrove con un indice posizionale, va aggiornato.
  return (valid_mae, valid_rmse, valid_r2, test_mae, test_rmse, test_r2,
          valid_true_mm, valid_pred_mm, test_true_mm, test_pred_mm, history)

#LSTM
def build_lstm_model(input_dims, time_steps, lstm_units):
  inputs = Input(shape=(time_steps, input_dims))
  lstm_out = LSTM(lstm_units, activation='relu')(inputs)
  output = Dense(1)(lstm_out)
  model = Model(inputs=[inputs], outputs=output)
  return model

#CNN
def build_cnn_model(input_dims, time_steps, filters):
  inputs = Input(shape=(time_steps, input_dims))
  x = Conv1D(filters=filters, kernel_size=1, activation='relu')(inputs)
  attention_mul = Flatten()(x)
  output = Dense(1)(attention_mul)
  model = Model(inputs=[inputs], outputs=output)
  return model


def attention_3d_block(inputs):
  input_dim = int(inputs.shape[2])
  a = inputs
  a = Dense(input_dim, activation='softmax')(a)
  a_probs = Permute((1, 2))(a)
  output_attention_mul = Multiply()([inputs, a_probs])
  return output_attention_mul


def build_bilstm_cnn_attention_model(input_dims, time_steps, lstm_units):
  inputs = Input(shape=(time_steps, input_dims))

  lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(inputs)

  x2 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(lstm_out)

  x3 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x2)
  x3 = BatchNormalization()(x3)
  x3 = Conv1D(filters=64, kernel_size=3, padding='same')(x3)
  x3 = BatchNormalization()(x3)
  x3 = add([x2, x3])
  x3 = Activation('relu')(x3)

  attention_mul = attention_3d_block(x3)
  attention_mul = Flatten()(attention_mul)
  output = Dense(1)(attention_mul)

  model = Model(inputs=[inputs], outputs=output)
  return model