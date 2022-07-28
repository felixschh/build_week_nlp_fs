import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch import optim
# import sys
# sys.path.insert(0, 'C:/Users/asus/Documents/GitHub/toxic_behavior/model1.py')
from model import Classifier

from data_handler import test_loader, train_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


MAX_SEQ_LEN = 32
model = Classifier(MAX_SEQ_LEN, 300, 16, 16)
model.to(device)

criterion = nn.BCEWithLogitsLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.parameters(), lr=0.003)        


emb_dim = 300
epochs = 100
# print_every = 40
train_losses, test_losses, accuracies = [], [], []

for e in range(epochs):
    running_loss, running_test_losses, running_test_accuracy = 0, 0, 0
    # print(f"Epoch: {e+1}/{epochs}")

    for i, (sentences, labels) in enumerate(iter(train_loader)):
        sentences, labels = sentences.to(device), labels.to(device)
        sentences.resize_(sentences.size()[0], 32* emb_dim)
        
        optimizer.zero_grad()
        
        output = model.forward(sentences)   # 1) Forward pass
        train_loss = criterion(output, labels.float()) # 2) Compute loss
        train_loss.backward()                  # 3) Backward pass
        optimizer.step()                 # 4) Update model
        
        running_loss += train_loss.item()
        
        # if i % print_every == 0:
        #     print(f"\tIteration: {i}\t Loss: {running_loss/print_every:.4f}")
        #     running_loss = 0
    avg_running_loss = running_loss/len(train_loader)
    train_losses.append(avg_running_loss)
    corrects = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, (sentences_test, labels_test) in enumerate(iter(test_loader)):
            sentences_test, labels_test = sentences_test.to(device), labels_test.to(device)
            sentences_test.resize_(sentences_test.size()[0], 32* emb_dim)

            output_test = model.forward(sentences_test)
            test_loss = criterion(output_test, labels_test.float())

            running_test_losses += test_loss.item()

            # prediction_label = torch.argmax(output_test, dim=1)
            prediction_label = torch.sigmoid(output_test)
            # print(prediction_label)
            total += labels_test.size(0)

            # running_test_accuracy += torch.sum(prediction_label==labels_test) / len(labels_test)        # need to fix this line
            classes = prediction_label > 0.5
            result = torch.sum(classes == labels_test, dim= 1) == len(labels_test)
            running_test_accuracy = sum(result)/len(result)   
        avg_test_loss = running_test_losses/len(test_loader)
        test_losses.append(avg_test_loss)
        avg_running_accuracy = running_test_accuracy/len(test_loader)
        accuracies.append(avg_running_accuracy.item())

    model.train()

    print(f"Epoch: {e+1}/{epochs}, Train loss: {avg_running_loss:.4f}, Test loss: {avg_test_loss:.4f}, Accuracy: {avg_running_accuracy:.4f}" )

    if e % 5 == 0:
        torch.save(model.state_dict(), f'/Users/felixschekerka/Desktop/build_week_nlp_fs/neuralnetwork/model_states/trained_state_{e}.pt')
        torch.save(model.state_dict(), f'/Users/felixschekerka/Desktop/build_week_nlp_fs/neuralnetwork/model_states/trained_state_{e}.pth')


plt.plot(train_losses, label='Train loss')
plt.plot(test_losses, label='Test losses')
plt.plot(accuracies, label='Accuracy')
plt.legend()
plt.show()