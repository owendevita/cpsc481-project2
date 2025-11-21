from keras.callbacks import EarlyStopping
from keras import layers, models
import time
import tensorflow as tf
import random
import numpy as np

SEED = 333

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def mlp_model(x_train, y_train, x_val, y_val, x_test, y_test):
    mlp = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

    mlp.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=2,
        restore_best_weights=True
    )

    start_time = time.time()
    history = mlp.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=32,
        callbacks=[early_stop]
    )
    mlp_training_time = time.time() - start_time

    test_loss, test_acc = mlp.evaluate(x_test, y_test, verbose=0)
    
    print()
    print(f"MLP training time: {mlp_training_time:.2f}s")
    print(f"MLP validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"MLP test accuracy: {test_acc:.4f}")
    print()

    return mlp, history