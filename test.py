import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data.dataset import MyDataset
from model.informer import Lstm_Informer
from utils.setseed import set_seed
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

lr = 0.001
epochs = 10
batch_size = 512
seq_len = 64
label_len = 64
pred_len = 0


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(0)
    print('device:', device)

    Station = "PTIS"
    df = pd.read_csv(f"./data/{Station}.csv")
    # print('df:', df)

    old_train = df.iloc[: 78840-(seq_len-1), :]
    test = df.iloc[78840-(seq_len-1):, :]  #

    tra_rate = 0.9  #
    tra_size = int(tra_rate * len(old_train))
    train = old_train.iloc[:tra_size]
    val = old_train.iloc[tra_size:]

    print('train:', train)
    print('val:', val)
    print('test:', test)

    scaler = StandardScaler()
    scaler.fit(train.iloc[:, 1:5].values.reshape(len(train), -1))
    scaler1 = StandardScaler()
    scaler1.fit(train.iloc[:, 5].values.reshape(len(train), -1))

    data1 = df.iloc[:, 1:5].values.reshape(len(df), -1)
    data1 = scaler.transform(data1)

    data2 = df.iloc[:, 5].values.reshape(len(df), -1)
    data2 = scaler1.transform(data2)

    merged_data = np.concatenate((data1, data2), axis=1)
    scaler_df = pd.DataFrame(merged_data)

    trainset = MyDataset(train, scaler, scaler1, seq_len=seq_len, label_len=label_len, pred_len=pred_len)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)  # 记得改shuffle为True!!

    valset = MyDataset(val, scaler, scaler1, seq_len=seq_len, label_len=label_len, pred_len=pred_len)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)  # 应该是False

    testset = MyDataset(test, scaler, scaler1, seq_len=seq_len, label_len=label_len, pred_len=pred_len)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')

    model = Lstm_Informer().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\ntrain...")
    train_losses = []
    val_losses = []

    for e in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        losses = []
        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(trainloader):
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)

            pred0 = model(batch_x, batch_x_mark)

            pred = pred0[:, -1:, :].to(device)

            true = batch_y[:, -1:, :].to(device)

            loss = criterion(pred, true)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_train_loss /= len(trainloader)
        train_losses.append(epoch_train_loss)
        print("Epochs:", e, " || train loss: %.4f" % epoch_train_loss)

        model.eval()
        with torch.no_grad():
            epoch_val_loss = 0.0
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(valloader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)

                pred0 = model(batch_x, batch_x_mark)  #
                pred = pred0[:, -1:, :].to(device)
                true = batch_y[:, -1:, :].to(device)

                loss = criterion(pred, true)
                epoch_val_loss += loss.item()
            epoch_val_loss /= len(valloader)
            val_losses.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), f"./log/{Station}_lstm_enco_model.pth")
            print(f'Saved best model with validation loss: {best_val_loss:.4f}')

        torch.cuda.empty_cache()

    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f"./img/{Station}_lstm_enco_train_val_losses.png")
    # plt.show()

    print("\ntest...")
    model.load_state_dict(torch.load(f"./log/{Station}_lstm_enco_model.pth"))

    with torch.no_grad():
        test_losses = []
        predictions = []
        true_values = []

        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(testloader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)

            pred0 = model(batch_x, batch_x_mark)
            pred = pred0[:, -1:, :].to(device)
            true = batch_y[:, -1:, :].to(device)

            loss = criterion(pred, true)
            test_losses.append(loss.item())
            predictions.append(pred.cpu().numpy())
            true_values.append(true.cpu().numpy())

    plt.figure(figsize=(10, 5))
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Batch Index')
    plt.ylabel('Loss')
    plt.title('Test Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./img/{Station}_lstm_enco_test_losses.png")
    # plt.show()

    print('mean test loss:', sum(test_losses) / len(test_losses))


    predictions = np.concatenate(predictions, axis=0)
    true_values = np.concatenate(true_values, axis=0)

    preds_result = predictions.reshape(-1, 1)

    trues_result = true_values.reshape(-1, 1)

    preds_result = scaler1.inverse_transform(preds_result)
    trues_result = scaler1.inverse_transform(trues_result)

    rmse = np.sqrt(mean_squared_error(preds_result, trues_result))  #
    bias = np.mean(preds_result - trues_result)
    mae = mean_absolute_error(preds_result, trues_result)
    r_squared = r2_score(trues_result, preds_result)

    diff_pred_result = preds_result.flatten()

    diff_trues_result = trues_result.flatten()
    diff_df = pd.DataFrame({'Pred_diff': diff_pred_result, 'True_diff': diff_trues_result})
    diff_df.to_csv(f"./log/{Station}_result.csv", index=False)

