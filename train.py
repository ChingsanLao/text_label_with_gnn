import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from networks import GraphContourLabeller
from datasets import GraphImageDataset
from utils import read_config_file

class GraphContourLabelerTrainer:
    def __init__(self, config_path):

        self.config = read_config_file(config_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = GraphContourLabeller().to(self.device)

        self.optimizer = Adam(
            self.model.parameters(), 
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        

        self.accuracy = MulticlassAccuracy(num_classes=18).to(self.device)
        self.f1_score = MulticlassF1Score(num_classes=18).to(self.device)
        

        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, train_loader):
        """
        Train the model for one epoch.
        
        Args:
        - train_loader (DataLoader): Training data loader
        
        Returns:
        - Dict containing epoch metrics
        """
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        total_f1 = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()
            
            outputs = self.model(batch)

            loss = self.criterion(outputs, batch.y)
            
            loss.backward()
            self.optimizer.step()
            
    
            total_loss += loss.item()
            total_accuracy += self.accuracy(outputs, batch.y.argmax(dim=-1))
            total_f1 += self.f1_score(outputs, batch.y.argmax(dim=-1))
        

        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader)
        avg_f1 = total_f1 / len(train_loader)
        
        return {
            'loss': avg_loss, 
            'accuracy': avg_accuracy, 
            'f1_score': avg_f1
        }

    
    def train(self, train_dataset, val_dataset=None):
        """
        Train the model.
        
        Args:
        - train_dataset (GraphImageDataset): Training dataset
        - val_dataset (GraphImageDataset, optional): Validation dataset
        """

        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.get('batch_size', 16), 
            shuffle=True
        )

        best_f1_score = 0
        best_model_path = "best_model.pth"

        epoch = 0
        while True:
    
            train_metrics = self.train_epoch(train_loader)
    

            if train_metrics['f1_score'] > best_f1_score:
                best_f1_score = train_metrics['f1_score']
                torch.save(self.model.state_dict(), best_model_path)
                print("best model saved")

    
            print(f"Epoch {epoch + 1}")
            print("Training Metrics:", train_metrics)
            print("-" * 50)
    
            epoch += 1


def main():
    image_dir = "original_images"
    json_dir = "dataset"
    config_path = "config.json"
    train_dataset = GraphImageDataset(image_dir, json_dir)
    trainer = GraphContourLabelerTrainer(config_path)
    trainer.train(train_dataset)

if __name__ == "__main__":
    main()