from glob import glob
import numpy as np
import _pickle as pickle
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style(style='white')

# obtain the paths for the saved model history
all_pickles = sorted(glob("results/*.pickle"))
# extract the name of each model
model_names = [item[8:-7] for item in all_pickles]


# extract the accuracy history for each model
valid_accuracy = [pickle.load(open(i, "rb"))['val_accuracy'] for i in all_pickles]
train_accuracy = [pickle.load(open(i, "rb"))['accuracy'] for i in all_pickles]

# save the number of epochs used to train each model
num_epochs = [len(valid_accuracy[i]) for i in range(len(valid_accuracy))]

fig = plt.figure(figsize=(16, 5))


# plot the training accuracy vs. epoch for each model
ax3 = fig.add_subplot(121)
for i in range(len(all_pickles)):
    ax3.plot(np.linspace(1, num_epochs[i], num_epochs[i]),
             train_accuracy[i], label=model_names[i])
# clean up the plot
ax3.legend()
ax3.set_xlim([1, max(num_epochs)])
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')

# plot the validation accuracy vs. epoch for each model
ax4 = fig.add_subplot(122)
for i in range(len(all_pickles)):
    ax4.plot(np.linspace(1, num_epochs[i], num_epochs[i]),
             valid_accuracy[i], label=model_names[i])
# clean up the plot
ax4.legend()
ax4.set_xlim([1, max(num_epochs)])
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')

plt.show()
