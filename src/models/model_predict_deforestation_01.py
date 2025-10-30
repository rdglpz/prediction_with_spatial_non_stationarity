from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.callbacks import EarlyStopping


def model01(X_tr, Y_tr, X_va, Y_va):
    i = Input(shape = X_tr[0].shape)
    x = Conv2D(16, (3,3), activation = 'relu')(i)
    x = Dropout(0.2)(x)
    x = Conv2D(32, (6,6), strides = 1, activation = 'relu')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (12,12), strides = 1, activation = 'relu')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(50, activation = 'relu')(x)
    x = Dense(50, activation = 'relu')(x)
    x = Dense(1, activation = "sigmoid")(x)

    model = Model(i,x)
    model.summary()
    callback = EarlyStopping(monitor='loss', patience = 3)
    
    model.compile(optimizer = "adam", loss = "binary_crossentropy",
                  metrics = ['accuracy'])

    r = model.fit(X_tr, Y_tr, validation_data = (X_va, Y_va), epochs = 5,
                  callbacks = [callback], batch_size = int(len(X_tr)/100))
    
    return model
    
    

    
    