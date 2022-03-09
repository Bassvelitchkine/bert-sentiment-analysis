import torch
import torch.nn as nn

from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

from dataset import CustomDataset
from loader import CustomDataLoader
from model import BertClassifier


class Classifier:
    """
    The classifier that wraps all the objects. It's the wrapper for the whole project.
    """
    def __init__(self,
        tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'),
        dataset_builder=CustomDataset,
        loader_builder=CustomDataLoader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=AdamW,
        scheduler_builder=get_linear_schedule_with_warmup,
        model=BertClassifier,
        batch_size=5,
        epochs=2,
        **kwargs) -> None:

        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_ = model(**kwargs).to(self.device_)
        self.optimizer_ = optimizer(self.model_.parameters(), lr=1e-4, eps=1e-8)
        self.tokenizer_ = tokenizer
        self.criterion_ = criterion
        self.batch_size_ = batch_size
        self.epochs_ = epochs
        
        self.dataset_builder_ = dataset_builder
        self.loader_builder_ = loader_builder
        self.scheduler_builder_ = scheduler_builder

        self.category_to_id_mapper_ = None
        self.label_to_id_mapper_ = None

    def __instantiate(self, filepath, train=True):
        '''
        A private method to instantiate all of the objects needed:
        - optimizer
        - scheduler
        - dataset
        - ...
        '''
        self.dataset_ = self.dataset_builder_(filepath, self.tokenizer_, category_to_id_mapper=self.category_to_id_mapper_, label_to_id_mapper=self.label_to_id_mapper_)

        shuffle = True if train else False
        self.loader_ = self.loader_builder_(dataset=self.dataset_, shuffle=shuffle, batch_size=self.batch_size_).loader

        training_steps = len(self.loader_) * self.epochs_
        self.scheduler_ = self.scheduler_builder_(self.optimizer_, num_warmup_steps=4, num_training_steps=training_steps)

        self.category_to_id_mapper_ = self.dataset_.category_to_id_mapper
        self.label_to_id_mapper_ = self.dataset_.label_to_id_mapper

        self.id_to_category_mapper_ = self.dataset_.id_to_category_mapper
        self.id_to_label_mapper_ = self.dataset_.id_to_label_mapper

    def __run_training_epoch(self, epoch):
        '''
        A private method to run the model for an epoch
        '''
        total_loss = 0.
        total_steps = len(self.loader_)

        for step, (encoded_sentences, position_indexes, categories, labels) in enumerate(self.loader_):
            encoded_sentences = encoded_sentences.to(self.device_)
            position_indexes = position_indexes.to(self.device_)
            categories = categories.to(self.device_)
            labels = labels.to(self.device_)

            outputs = self.model_(encoded_sentences, position_indexes, categories)
            loss = self.criterion_(outputs, labels)

            self.optimizer_.zero_grad()
            loss.backward()
            self.optimizer_.step()
            self.scheduler_.step()

            total_loss += loss.item()

            if (step + 1) % 10 == 0:
                loss_to_display = total_loss / ((step + 1) * self.batch_size_)
                trailing_str = '\r' if step < total_steps - 1 else ''
                print(f"Epoch: [{epoch + 1}/{self.epochs_}] - Step: [{step+1}/{total_steps}] - Avg loss: {loss_to_display: .4f}", end=trailing_str)
    
    def __run_testing_epoch(self):
        '''
        A private method to run a testing epoch/evaluation epoch
        '''
        _ = self.model_.eval()

        correct = .0
        predictions = []
        
        with torch.no_grad():
            for encoded_sentences, position_indexes, categories, labels in self.loader_:
                encoded_sentences = encoded_sentences.to(self.device_)
                position_indexes = position_indexes.to(self.device_)      
                categories = categories.to(self.device_)
                labels = labels.to(self.device_)

                outputs = self.model_(encoded_sentences, position_indexes, categories)
                _, predicted = torch.max(outputs, 1)
                predictions.append(int(predicted))
                correct += (predicted == labels).sum().item()

        accuracy = correct / len(predictions) * 100
        print(f"Accuracy: {accuracy: .2f}%")

        return predictions

    def train(self, trainfile, devfile=None):
        """
        Trains the classifier model on the training set stored in file trainfile
        WARNING: DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        self.__instantiate(trainfile, train=True)

        _ = self.model_.train()
        for epoch in range(self.epochs_):
            self.__run_training_epoch(epoch)

        _ = self.model_.eval()
        with torch.no_grad():
            _ = self.__run_testing_epoch()

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        self.__instantiate(datafile, train=False)
        
        _ = self.model_.eval()
        with torch.no_grad():
            predictions = self.__run_testing_epoch()

        return [self.id_to_label_mapper_[prediction] for prediction in predictions]

if __name__ == "__main__":
    
    filepath = './data/traindata.csv'
    clf = Classifier()
    clf.train(filepath)

