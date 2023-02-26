import numpy as np
from keras.models import Model
from keras.layers import  Dropout,Flatten
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.layers import Input,Dense
from tensorflow.python.keras.utils.np_utils import to_categorical
import matplotlib
matplotlib.use('TkAgg')
import warnings
from keras.callbacks import EarlyStopping,ModelCheckpoint

warnings.filterwarnings('ignore')


Train = np.load(r'C:\Users\HP\Desktop\XX.npy')
Label = np.load(r'C:\Users\HP\Desktop\YY.npy')

label = to_categorical(Label)

def build_model_seq(x_train, y_train):
    inputt=Input(shape=(x_train.shape[1],x_train.shape[2]))
    output=Conv1D(filters=32, kernel_size=32,activation='relu')(inputt)
    output=MaxPooling1D(4)(output)
    output = Flatten()(output)
    output = Dense(128,activation='relu')(output)
    output = Dropout(0.5)(output)
    output = Dense(64,activation='relu')(output)
    output = Dropout(0.5)(output)
    output=Dense(2,activation='softmax')(output)
    model = Model(inputs=inputt, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def fit_model_seq(model,model_file, x_train, y_train,frac_val,epochs, batch_size,verbose):
    if y_train.ndim==1:
        y_train = to_categorical(y_train)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=5)
    mc = ModelCheckpoint(model_file, monitor='val_accuracy', mode='max', verbose=verbose, save_best_only=True)
    history=model.fit(x_train, y_train, validation_split=frac_val,epochs=epochs, batch_size=batch_size, verbose=verbose,callbacks=[es, mc])
    return history

if __name__ == '__main__':
    model_file = 'hgmd' + str(10) + '.seq.h5'
    pretrain_model = build_model_seq(Train,label)
    history = fit_model_seq(pretrain_model, model_file, Train, label, 0.2, 50, 128,verbose=1)

