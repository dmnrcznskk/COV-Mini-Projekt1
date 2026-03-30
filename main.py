import torch
import torch.nn as nn
import torch.optim as optim

from src.data import get_loaders
from src.models.petsv1 import PetsCNNv1
from src.models.petsv2 import PetsCNNv2
from src.models.petsv3 import PetsCNNv3
from src.models.petsv4 import PetsCNNv4
from src.train_and_eval import train_model
from src.visualize import visualize_training

def main():
    EPOCHS = 100
    LR = 0.001
    BATCH_SIZE = 64


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_loaders(batch_size=BATCH_SIZE)

    model = PetsCNNv4(num_classes=37).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    model_name="PetsCNNv4_100_scheduler_wd"

    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=EPOCHS,
        model_name=model_name
    )
    
    print("Trening zakonczony")
    visualize_training(history, model_name=model_name)


if __name__=="__main__":
    main()


