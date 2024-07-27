import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score,roc_auc_score
from tqdm import tqdm
import pickle

def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)

# def evaluate_thr(labels, scores, step,adj=True):
#     min_score = min(scores)
#     max_score = max(scores)
#     best_f1 = 0.0
#     best_preds = []
#     # 在max和min之间隔开为1000个数，迭代用数值作为阈值计算f1
#     for th in tqdm(np.linspace(min_score, max_score, step), ncols=70):
#         preds = (scores > th).astype(int)
#         if adj:
#             preds = adjust_predicts(labels, preds)
#             # print("preds",preds.dtype)
#             # print("labels",labels.dtype)
#         f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
#         if f1 > best_f1:
#             best_f1 = f1
#             best_preds = preds
#
#     return best_preds

def evaluate(labels, scores, step, adj=True):
    # best f1
    min_score = min(scores)
    max_score = max(scores)
    best_f1 = 0.0
    best_preds = []
    # 在max和min之间隔开为1000个数，迭代用数值作为阈值计算f1
    for th in tqdm(np.linspace(min_score, max_score, step), ncols=70):
        preds = (scores > th).astype(int)
        if adj:
            preds = adjust_predicts(labels, preds)
            # print("preds",preds.dtype)
            # print("labels",labels.dtype)
        f1 = f1_score(y_true=labels, y_pred=preds,average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_preds = preds

    return best_preds


def adjust_predicts(label, pred=None):
    predict = pred.astype(bool)
    actual = label > 0.1
    anomaly_state = False
    for i in range(len(label)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
        elif not actual[i]:
            anomaly_state = False

        if anomaly_state:
            predict[i] = True
    return predict.astype(int)

def eval_ad_result(test_pred_list, test_labels_list):
    f1 = f1_score(test_labels_list, test_pred_list, average='macro')
    precise = precision_score(test_labels_list, test_pred_list, average='macro')
    recall = recall_score(test_labels_list,test_pred_list, average='macro')
    return f1,precise,recall


def train_model(model, train_dataloader, optimizer,device):
    model.train()
    criterion = nn.MSELoss()
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_loss = 0
    masked_ratio_train=0.8

    for inputs in train_dataloader:
        inputs = inputs[0]
        original_t=inputs
        original_t=original_t.to(device)
        spectrum = np.fft.fft(inputs)
        amplitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        amplitude = torch.tensor(amplitude, dtype=torch.float32).to(device)
        phase = torch.tensor(phase, dtype=torch.float32).to(device)
        optimizer.zero_grad()
        inputs=inputs.to(device)
        reconstructed_tem,reconstructed_fre = model(inputs,amplitude,phase,masked_ratio_train)

        loss1=criterion(reconstructed_tem,original_t)
        loss2=criterion(reconstructed_fre,original_t)
        loss=loss1+loss2
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss=total_loss / len(train_dataloader)
    return avg_loss


def train_model_without_Fre(model, train_dataloader, optimizer, device):
    model.train()
    criterion = nn.MSELoss()
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_loss = 0
    masked_ratio_train = 0.0

    for inputs in train_dataloader:
        inputs = inputs[0]
        original_t = inputs
        original_t = original_t.to(device)
        optimizer.zero_grad()
        inputs = inputs.to(device)
        reconstructed_tem = model(inputs, masked_ratio_train)

        loss= criterion(reconstructed_tem, original_t)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    return avg_loss

def anomaly_detection(model,test_dataloader,num_features,device):
    model.eval()
    all_score=torch.randn(0).to(device)

    with torch.no_grad():
        for inputs in test_dataloader:
            inputs = inputs[0]
            targets=inputs.reshape(-1,num_features)
            targets=targets.to(device)
            spectrum = np.fft.fft(inputs)
            amplitude = np.abs(spectrum)
            phase = np.angle(spectrum)
            amplitude = torch.tensor(amplitude, dtype=torch.float32).to(device)
            phase = torch.tensor(phase, dtype=torch.float32).to(device)
            inputs=inputs.to(device)

            masked_ratio_test = 0.0
            reconstructed_tem, reconstructed_fre = model(inputs, amplitude, phase,masked_ratio_test)
            # score=torch.mean((targets-preds)**2,dim=1)
            reconstructed_tem=reconstructed_tem.reshape(-1,num_features)
            reconstructed_fre = reconstructed_fre.reshape(-1, num_features)
            # score_tem=torch.mean((targets-reconstructed_tem)**2,dim=1)
            score_tem = torch.sub(targets, reconstructed_tem)
            # score_fre = torch.mean((targets - reconstructed_fre) ** 2, dim=1)
            score_fre=torch.sub(targets,reconstructed_fre)
            anomaly_score=score_tem+score_fre
            score=anomaly_score.sum(axis=1)
            all_score = torch.cat((all_score, score)).to(device)

        all_score = all_score.cpu().detach().numpy()

    return all_score

def anomaly_detection_without_Fre(model,test_dataloader,num_features,device):
    model.eval()
    all_score=torch.randn(0).to(device)

    with torch.no_grad():
        for inputs in test_dataloader:
            inputs = inputs[0]
            targets=inputs.reshape(-1,num_features)
            targets=targets.to(device)
            inputs=inputs.to(device)

            masked_ratio_test = 0.0
            reconstructed_tem = model(inputs,masked_ratio_test)
            reconstructed_tem=reconstructed_tem.reshape(-1,num_features)

            score_tem=torch.sub(targets,reconstructed_tem)
            anomaly_score=score_tem
            score=anomaly_score.sum(axis=1)
            all_score = torch.cat((all_score, score)).to(device)

        all_score = all_score.cpu().detach().numpy()

    return all_score
