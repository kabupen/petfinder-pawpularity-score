
import torch

from src.helper import *
from src.model import *
from src.dataset import *
from src.augmentations import *

if __name__ == "__main__":

    CSV_PATH = '../input/petfinder-pawpularity-score/'
    IMG_PATH = '../input/petfinder-pawpularity-score/train/'

    device = torch.device('cuda' if torch.cuda.is_avaiable() else 'cpu')
    print(f'Using device : {device}')

    fix_random_seed()

    #----------------------------
    # Dataset/Dataloader
    #----------------------------
    df_train = pd.read_csv(f'{CSV_PATH}/train.csv')
    df_test = pd.read_csv(f'{CSV_PATH}/test.csv')
    X_train, X_test, y_train, y_test = train_test_split(df_train['Id'], df_train.iloc[:, 1:], test_size=0.3)

    train_dataset = PetfinderDataset()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = PetfinderDataset()
    test_loader = DataLoader(test_loader, batch_size=32, shuffle=False)


    #----------------------------
    # model
    #----------------------------

    model = PetfinderModel(
            model_name = "swin_large_patch4_window12_384",
            out_features = 1,
            in_chans = 3,
            pretrained = True,
            num_dense = 12)
    model = model.to(device) 
    print(model)


    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.MSELoss()


    #----------------------------
    # train loop
    #----------------------------
    loss = {'train':[], 'val':[]}
    for epoch in range(EPOCH):

        ### train
        runnning_loss = 0.
        model.train()
        for batch_idx, (imgages, dense, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            images, dense = images.to(device), dense.to(device)

            outputs = model(images, dense)

            loss = torch.sqrt(criterion(outputs, labels))

            loss.backward()
            optimizer.step()

            runnning_loss += loss.item()
        
        runnning_loss['train'].append(runnning_loss)


        ### evaluation
        model.eval()
        runnning_loss = 0.
        with torch.no_grad():
            for batch_idx, (imgages, dense, labels) in enumerate(test_loader):
    
                images, dense = images.to(device), dense.to(device)
    
                outputs = model(images, dense)
    
                loss = torch.sqrt(criterion(outputs, labels))
    
                 
            runnning_loss['val'].append(runnning_loss)
        
        print(runnning_loss)

    return
