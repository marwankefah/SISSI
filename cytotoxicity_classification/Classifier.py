import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2)
                                  for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x, device="cpu"):
        ''' Forward pass through the network, returns the output logits '''
        x = x.view(x.shape[0], -1)
        x = x.type(torch.FloatTensor).to(device)
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        return F.log_softmax(self.output(x), dim=1)

    def loadModel(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        return


def savemodel(model, path):
    checkpoint = {'input_size': model.hidden_layers[0].in_features,
                  'output_size': model.output.out_features,
                  'hidden_layers': [each.out_features for each in model.hidden_layers],
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, path)


def load_checkpoint(filepath, lr=0.001, dropout=0.2, device="cpu"):
    checkpoint = torch.load(filepath, map_location=device)
    model, optimizer, criterion = defineModel(
        checkpoint['input_size'], checkpoint['output_size'], checkpoint['hidden_layers'], dropout=dropout, lr=lr, device=device)
    model.load_state_dict(checkpoint['state_dict'])

    return model, optimizer, criterion


def validation(model, testloader, criterion, device="cpu"):
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        accuracy = accuracy / len(testloader)
        test_loss = test_loss / len(testloader)
    return test_loss, accuracy


def training(model, trainloader, validloader, criterion, optimizer, modelPath, device="cpu", epochs=10):
    valid_losses = []
    best_acc = 0
    for e in range(epochs):
        running_loss = 0
        model.train()
        for images, labels in trainloader:

            optimizer.zero_grad()

            images, labels = images.to(device), labels.to(device)
            out = model.forward(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            model.eval()
            valid_loss = 0
            valid_accuracy = 0
            with torch.no_grad():
                valid_loss, valid_accuracy = validation(
                    model, validloader, criterion, device)
                valid_losses.append(valid_loss / len(validloader))
            print("epoch {0}".format(e))
            print("Training loss = {0} ".format(
                running_loss / len(trainloader)))
            print("validation loss = {0} ".format(
                valid_loss / len(validloader)))
            print("Test accuracy = {0} % \n".format(valid_accuracy * 100))
            if best_acc < (valid_accuracy * 100):
                best_acc = (valid_accuracy * 100)
                savemodel(model, modelPath)

    return valid_loss


def defineTransforms():
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Grayscale(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Grayscale(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    return train_transforms, test_transforms


def predictforImage(model, img, device):
    model.eval()
    with torch.no_grad():
        logps = model.forward(img, device)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        return top_p, top_class


def defineModel(input, output, hidden_layers, dropout=0.2, lr=0.001, device="cpu"):
    model = Classifier(input_size=input, output_size=output,
                       hidden_layers=hidden_layers, drop_p=dropout)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    return model, optimizer, criterion
