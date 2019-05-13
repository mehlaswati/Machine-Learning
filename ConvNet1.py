#Author : Swati
#Best network

import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import torch.utils.data as data
import torchvision.transforms as transforms
from pytorch_nsynth.nsynth import NSynth
import numpy as np
import torch
import copy
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.ticker as ticker
import itertools
plt.rcParams.update({'figure.max_open_warning': 0})

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

EPOCHS = 30


# Defining the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 6,3)  # in channles, out channels, kernel size
        self.pool = nn.MaxPool1d(5, stride = 5)  # kernel size and stride
        self.conv2 = nn.Conv1d(6, 16,3)
        self.conv3 = nn.Conv1d(16, 20, 5)
        self.conv2_bn = nn.BatchNorm1d(16)
        self.conv3_bn = nn.BatchNorm1d(20)
        self.dense1_bn = nn.BatchNorm1d(9220)
        self.dense2_bn = nn.BatchNorm1d(260)
        self.fc1 = nn.Linear(12780, 9220)  # in features, out features
        self.fc2 = nn.Linear(9220, 260)
        self.fc3 = nn.Linear(260, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(self.conv2_bn(F.relu(self.conv2(x))))
        x = self.pool(self.conv3_bn(F.relu(self.conv3(x))))
        x = x.view(-1, self.num_flat_features(x))  # view changes the dimensionality of the Tensor
        x = self.dense1_bn(F.relu(self.fc1(x)))
        x = self.dense2_bn(F.relu(self.fc2(x)))
        x = F.softmax(self.fc3(x), dim=1)
        
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


#####################################################################################################

net = Net()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.5)


#####################################################################################################

def plotCurve(saved_train_losses, saved_validation_losses):
    # This function plots the training and validation curves
    fig, ax = plt.subplots()

    x = np.linspace(1, EPOCHS, EPOCHS)
    saved_validation_losses = np.array(saved_validation_losses)
    saved_train_losses = np.array(saved_train_losses)

    ax.set_title("Average Model Loss over Epochs")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Average Loss")

    # Adjust x-axis ticks
    tick_spacing = 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    ax.plot(x, saved_train_losses, color='purple', label="Training Loss")
    ax.plot(x, saved_validation_losses, color='red', label="Validation Loss")
    plt.legend(loc="upper right")
    fig.savefig('./model_loss.png')


# Train the network
def train(train_loader, valid_loader, sizeOfTrainDataset, sizeOfValidDataset):
    best_model_wts = copy.deepcopy(net.state_dict())
    saved_train_losses = []
    saved_loss_valid = []
    best_acc = 0.0
    classesPlt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for epoch in range(EPOCHS):

        ##############------Train ---------##################################################3

        train_loss = 0.0

        for samples, instrument_family_target, instrument_source_target, targets in train_loader:
            inputs = samples
            # print('Shape Before', inputs.shape)
            inputs = samples[:, None]
            labels = instrument_family_target

            for i in range(len(labels)):
                label = labels[i]
                if label.item() == 10:
                    label = 9

                if classesPlt[label] < 2:
                    t = inputs[i][0, :].numpy()
                    classesPlt[label] += 1

            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimizer
            outputs = net(inputs.float())

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            # Print statistics
            train_loss += loss.item()

        saved_train_losses.append(train_loss / sizeOfTrainDataset)
        print('Epoch ' + str(epoch + 1) + ', Training loss: ' + str(train_loss / sizeOfTrainDataset))

        print('Finished Training train dataset')

        ############----------------Validation-----------###################

        correct = 0
        total = 0
        validation_loss = 0.0

        with torch.no_grad():
            for samples, instrument_family_target, instrument_source_target, targets in valid_loader:
                inputs = samples[:, None]
                labels = instrument_family_target

                inputs, labels = inputs.to(device), labels.to(device)

                validation_outputs = net(inputs.float())

                loss = criterion(validation_outputs, labels)

                _, predicted = torch.max(validation_outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                validation_loss += loss.item()

            print('Epoch ' + str(epoch + 1) + ', Validation Set loss: ' + str(validation_loss / sizeOfValidDataset))
            saved_loss_valid.append(validation_loss / sizeOfValidDataset)

        epoch_acc = correct / total
        print('Accuracy of validation for Epoch ' + str(epoch + 1) + ' :%d %%\n' % (100 * epoch_acc))
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(net.state_dict())

    print('Finished Training')

    # Uncomment to plot Learning Curve
    plotCurve(saved_train_losses, saved_loss_valid)
    net.load_state_dict(best_model_wts)
    torch.save(net.state_dict(), "./paramQ2")



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    # This function plots a confusion marix given a 2D confusion matrix
    fig, ax = plt.subplots()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig.savefig('./Confusion matrix')


def plotWaveform(samples, title, saveAs):
    # This function plots a sample waveform
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.plot(samples)
    fig.savefig(saveAs)

def findLargest(item_list):
    # This function finds the first and second largest values from a list of tensors
    first = 0.0
    second = 0.0
    first_class = 0
    second_class = 0
    for i in range(len(item_list)):    
        if item_list[i].item() > first:
            second = first          
            first = item_list[i].item()            
            second_class = first_class
            first_class = i
        elif item_list[i].item() > second and second != first  :
            second = item_list[i].item()
            second_class = i
    return first, second, first_class, second_class

def test(test_loader):
    # This function performs testing on the  saved best network
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    conf = [[0 for _ in range(10)] for __ in range(10)]

    correct = 0
    total = 0

    with torch.no_grad():
        for samples, instrument_family_target, instrument_source_target, targets in test_loader:
            inputs = samples[:, None]
            labels = instrument_family_target

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs.float())

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    finalAccuracy = (100 * correct / total)
    print('\nAccuracy of the network on test data is: %d %%\n' % finalAccuracy)

    max_prob = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    samples_best = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    least_prob = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    class_1_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    decision_sample = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    class_labels_correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    class_labels_wrong = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    classes = {0: 'Bass', 1: 'brass', 2: 'flute', 3: 'guitar', 4: 'keyboard', 5: 'mallet', 6: 'organ', 7: 'Reed',
               8: 'string', 9: 'vocal'}

    with torch.no_grad():
        for samples, instrument_family_target, instrument_source_target, targets in test_loader:
            inputs = samples[:, None]
            labels = instrument_family_target

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs.float())

            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1
                conf[predicted[i].item()][label] += 1
                if predicted[i].item() == label:

                    # THE LOGIC TO SHOW CLASS WITH HIGH PROBABILITY
                    tempProbability = outputs[i][label]
                    if tempProbability > max_prob[label]:
                        max_prob[label] = tempProbability
                        samples_best[label] = inputs[i][0]

                    first, second, first_class, second_class = findLargest(outputs[i])
                    if (first - second) < least_prob[first_class]:
                        least_prob[first_class] = (first - second)
                        class_1_2[first_class] = second_class
                        decision_sample[first_class] = inputs[i][0]                          

                if label == predicted[i].item() and class_labels_correct[label] == 0:
                    class_labels_correct[label] = 1
                    plotWaveform( samples[i][:].cpu().numpy(), "Predicted = " + classes[(predicted[i].item())], "./Correct class "+classes[label])


                if label != predicted[i].item() and class_labels_wrong[label] == 0:
                    class_labels_wrong[label] = 1
                    text = "Predicted = " + classes[predicted[i].item()] + " Actual = " + classes[label]
                    saveAs = "./Correct class predicted-actual "
                    plotWaveform(samples[i][:].cpu().numpy(),text ,saveAs )


    

    for i in range(10):
        for j in range(10):
            print(str(conf[i][j]) + " ", end='')
        print()
    for items in classes.keys():
        print('Accuracy of %5s : %2d %%' % (
            classes[items], 100 * class_correct[items] / class_total[items]))

    conf = np.array(conf)
    plot_confusion_matrix(np.matrix(conf), classes=classes.items(), title='Confusion matrix, without normalization')

    for i in range(len(samples_best)):
        if not isinstance(samples_best[i], int):
            sampToPlot = samples_best[i][:].cpu().numpy()
            plotWaveform(sampToPlot,
                         'Best Prob waveform for Correct Class: ' + str(i) + 'with prob ' + str(max_prob[i].item()),
                         'best_prob_class' + str(i))
        if not isinstance(decision_sample[i], int):            
            dec_sample = decision_sample[i][:].cpu().numpy()            
            plotWaveform(dec_sample,'Correct Class: ' + str(i) + ', Prob diff %.5f' %  least_prob[i] + ' greater than Class: ' + str(class_1_2[i]),'best_prob_class_dec' + str(i))  
    print(max_prob)
    print(least_prob)


def main():
    # Subsampling
    subsample_transform = transforms.Lambda(lambda x: x[::4])
    toFloat = transforms.Lambda(lambda x: x / np.iinfo(np.int16).max)

    trainData = NSynth("/local/sandbox/nsynth/nsynth-train",
                       transform=transforms.Compose([subsample_transform, toFloat]),
                       blacklist_pattern=["synth_lead"],
                       categorical_field_list=["instrument_family", "instrument_source"])

    validation_dataset = NSynth("/local/sandbox/nsynth/nsynth-valid",
                                transform=transforms.Compose([subsample_transform, toFloat]),
                                blacklist_pattern=["synth_lead"],
                                categorical_field_list=["instrument_family", "instrument_source"])

    test_dataset = NSynth("/local/sandbox/nsynth/nsynth-test",
                          transform=transforms.Compose([subsample_transform, toFloat]),
                          blacklist_pattern=["synth_lead"],
                          categorical_field_list=["instrument_family", "instrument_source"])
    print(len(trainData))
    train_loader = data.DataLoader(trainData, batch_size=64, shuffle=True)

    valid_loader = data.DataLoader(validation_dataset, batch_size=64, shuffle=True)

    test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    train(train_loader, valid_loader, len(trainData), len(validation_dataset))
    test(test_loader)


main()
# 