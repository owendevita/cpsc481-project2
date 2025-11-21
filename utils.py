import numpy as np
import matplotlib.pyplot as plt
import keras
import mnist_cnn
import mnist_mlp
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns



def get_data():
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data("mnist.npz")

    x_train = x_train_full[:50000]
    y_train = y_train_full[:50000]

    x_val = x_train_full[50000:]
    y_val = y_train_full[50000:]

    print()
    print("SHAPES REPORT:")
    print("------------------------------------------------------------------------------")
    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("x_val:", x_val.shape)
    print("y_val:", y_val.shape)
    print("x_test:", x_test.shape)
    print("y_test:", y_test.shape)
    print()
    return x_train, y_train, x_val, y_val, x_test, y_test


def show_sample_images(x_train, y_train):
    plt.figure(num="5x5 Sample Images",figsize=(6, 6))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(x_train[i], cmap='gray')
        plt.title(f"True Label = {str(y_train[i])}", fontsize=10)
        plt.axis("off")

    plt.tight_layout(pad=2.0)

def plot_history(history, model_name="Model"):
    plt.figure(num=f"{model_name} Loss/Accuracy History", figsize=(12,4))
    
    # Loss
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    

def evaluate_model(model, x_test, y_test, is_mlp=False):
    y_pred_probs = model.predict(x_test)
    y_pred = y_pred_probs.argmax(axis=1)

    print()
    if is_mlp:
        print("MLP CLASSIFICATION REPORT:")
        print("------------------------------------------------------------------------------")
    else:
        print("CNN CLASSIFICATION REPORT:")
        print("------------------------------------------------------------------------------")
    
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    if is_mlp:
        plt.figure(num="MLP Confusion Matrix", figsize=(10,8))
    else:
        plt.figure(num="CNN Confusion Matrix",figsize=(10,8))
        
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    mis_idx = np.where(y_pred != y_test)[0]
    if len(mis_idx) > 0:
        if is_mlp:
            plt.figure(num="MLP Misclassified Examples", figsize=(12,5))
        else:
            plt.figure(num="CNN Misclassified Examples", figsize=(12,5))
        for i, idx in enumerate(mis_idx[:10]):
            plt.subplot(2,5,i+1)
            img = x_test[idx]
            if is_mlp:
                img = img.reshape(28,28)
            plt.imshow(img.squeeze(), cmap='gray')
            plt.title(f"T:{y_test[idx]} P:{y_pred[idx]}\nConf:{y_pred_probs[idx][y_pred[idx]]:.2f}", fontsize=12)
            plt.axis("off")
        plt.tight_layout(pad=1.5)


x_train, y_train, x_val, y_val, x_test, y_test = get_data()
show_sample_images(x_train, y_train)

x_train = x_train.astype("float32") / 255.0
x_val   = x_val.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0


x_train_mlp = x_train.reshape((x_train.shape[0], 784))
x_val_mlp = x_val.reshape((x_val.shape[0], 784))
x_test_mlp = x_test.reshape((x_test.shape[0], 784))

x_train_cnn = x_train[..., np.newaxis]
x_val_cnn = x_val[..., np.newaxis]
x_test_cnn = x_test[..., np.newaxis]

model_mlp, history_mlp = mnist_mlp.mlp_model(x_train_mlp, y_train, x_val_mlp, y_val, x_test_mlp, y_test)
model_cnn, history_cnn = mnist_cnn.cnn_model(x_train_cnn, y_train, x_val_cnn, y_val, x_test_cnn, y_test)

plot_history(history_mlp, "MLP")
plot_history(history_cnn, "CNN")

evaluate_model(model_mlp, x_test_mlp, y_test, is_mlp=True)
evaluate_model(model_cnn, x_test_cnn, y_test)

plt.show(block=True)
