import keras
import matplotlib.pyplot as plt


class PerformancePlotCallback(keras.callbacks.Callback):
    def __init__(self, x_test, y_test, model_name):
        self.x_test = x_test
        self.y_test = y_test
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs={}):

        y_pred = self.model.predict(self.x_test)
        fig, ax = plt.subplots(figsize=(8, 4))
        plt.scatter(self.y_test, y_pred, alpha=0.6,
                    color='#FF0000', lw=1, ec='black')

        lims = [0, 2]

        plt.plot(lims, lims, lw=1, color='#0000FF')
        plt.ticklabel_format(useOffset=False, style='plain')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim(lims)
        plt.ylim(lims)

        plt.tight_layout()
        plt.title(f'Prediction Visualization Keras Callback - Epoch: {epoch}')
        plt.savefig('model_train_images/' + self.model_name + "_" + str(epoch))
        plt.close()