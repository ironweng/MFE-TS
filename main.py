import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
from CFTSAD import CFTSAD_Model,Model_without_Fre,Model_without_Spa,Model_without_Comb
from base_function import train_model,anomaly_detection,evaluate,eval_ad_result,pkl_save
from base_function import train_model_without_Fre,anomaly_detection_without_Fre

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset',type=str, help='The dataset name')
    parser.add_argument('subseq_len',type=int, help='The number of subsequence')
    parser.add_argument('batch_size',default=32, type=int, help='The batch size')
    parser.add_argument('lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--task_type', type=str, choices=['normal','without_fre','without_spa','without_comb'],required=True, help='Ablation conditions')
    # parser.add_argument('mask', type=bool, default=None, help='whether mask tem_sequence')
    args = parser.parse_args()

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Loading data... ', end='')


    # if args.dataset=='SMD':
    #     data_train = np.loadtxt("D:/st_zp/CFTSAD/datasets/SMD/SMD_subset/train/machine-1-1.txt",delimiter=",")
    #     data_test = np.loadtxt("D:/st_zp/CFTSAD/datasets/SMD/SMD_subset/test/machine-1-1.txt",delimiter=",")
    #     label_test = np.loadtxt("D:/st_zp/CFTSAD/datasets/SMD/SMD_subset/test_label/machine-1-1.txt",delimiter=",")
    if args.dataset=='SMD':
        data_train = np.load("/datasets/SMD/SMD_subset/train/SMD_train.npy",)
        data_test = np.load("/datasets/SMD/SMD_subset/test/SMD_test.npy")
        label_test = np.load("/datasets/SMD/SMD_subset/test_label/SMD_test_label.npy")
    elif args.dataset=='MSL':                          
        data_train = np.load("/datasets/MSL/MSL_train.npy")
        data_test = np.load("/datasets/MSL/MSL_test.npy")
        label_test = np.load("/datasets/MSL/MSL_test_label.npy")
    elif args.dataset=='SMAP':                        
        data_train = np.load("/datasets/SMAP/SMAP_train.npy")
        data_test = np.load("/datasets/SMAP/SMAP_test.npy")
        label_test = np.load("/datasets/SMAP/SMAP_test_label.npy")
    elif args.dataset=='SWAT':                 
        train = np.load("/datasets/SWAT/SWaT_train.npy")
        test = np.load("/datasets/SWAT/SWaT_test.npy")
        label_test = np.load("/datasets/SWAT/SWaT_labels.npy",allow_pickle=True)
        label_test=label_test.astype(np.int32)
        # min_train=np.min(data_train,axis=0)
        # max_train=np.max(data_train,axis=0)
        # min_test=np.min(data_test,axis=0)
        # max_test = np.max(data_test,axis=0)
        # data_train=(data_train-min_train)/(max_train-min_train)
        # data_test=(data_test-min_test)/(max_test-min_test)
        invalid_mask1=np.isnan(train)|np.isinf(train)
        train[invalid_mask1]=0.0
        mean1=np.mean(train,axis=0)
        std1=np.std(train,axis=0) #计算每一列的标准差
        std1[std1==0.0]=1.0
        data_train=(train-mean1)/std1
        invalid_mask2 = np.isnan(test) | np.isinf(test)
        test[invalid_mask2] = 0.0
        mean2 = np.mean(test, axis=0)
        std2 = np.std(test, axis=0)
        std2[std2 == 0.0] = 1.0
        data_test = (test - mean2) / std2
    elif args.dataset=='WADI':
        data_train = np.load("/datasets/WADI/wadi_train.npy")
        data_test = np.load("/datasets/WADI/wadi_test.npy")
        label_test = np.load("/datasets/WADI/wadi_labels.npy")
        # has_invalid = np.any((label_test != 0) & (label_test != 1))
        # if has_invalid:
        #     print("包含无效值")
        # else:
        #     print("只含0和1")
    print("train_sets",data_train.shape)
    print("test_sets", data_test.shape)

    # mask_ratio=0.2
    #train datasets processing
    sequence_length = data_train.shape[0]
    num_features = data_train.shape[1]
    # print("num_features",num_features)

    train_data = torch.FloatTensor(data_train).unsqueeze(0)  # 添加额外的维度
    batch_size = args.batch_size
    subsequence_length = args.subseq_len
    subsequence_num = sequence_length // subsequence_length
    # print(subsequence_length_num)
    train_data = train_data[:, :subsequence_length * subsequence_num, :]
    # print("train-data",train_data.shape)
    train_data = train_data.view(-1, subsequence_length, num_features)  # 重塑数据以形成batch
    # print("train-data",train_data.shape)
    # 创建训练集的DataLoader
    train_dataset = TensorDataset(train_data)
    # print("train_dataset", len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # test datasets processing
    sequence_length_test = data_test.shape[0]
    subsequence_num_test = sequence_length_test // subsequence_length
    test_data = torch.FloatTensor(data_test).unsqueeze(0)  # 添加额外的维度

    test_data = test_data[:, :subsequence_length * subsequence_num_test, :]
    # print("train-data",train_data.shape)
    test_data = test_data.view(-1, subsequence_length, num_features)  # 重塑数据以形成batch
    # print("train-data",train_data.shape)
    # 创建训练集的DataLoader
    test_dataset = TensorDataset(test_data)
    # print("train_dataset", len(train_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    label_test = label_test[:subsequence_length * subsequence_num_test]

    task_type=args.task_type

    if task_type=='normal':
        model =CFTSAD_ Model(input_size=num_features,hidden_size=64,num_heads=4,num_layers=2,output_size=num_features).to(device)
        print("task_type=normal")
        optimezer=optim.Adam(model.parameters(),lr=args.lr)
        criterion=nn.MSELoss()
        model=model.to(device)

        num_epochs=args.epochs
        for epoch in range(num_epochs):
            train_loss=train_model(model, train_dataloader,optimezer,device)
            if (epoch+ 1)%10==0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss}")
#
        savemodel_path=f'D:/st_zp/CFTSAD/results/{args.dataset}/model_states/model_{args.dataset}_21.pth'
    # # save_path = 'D:/st_zp/Time-space correlation/results/model_states/model1.pkl'
        torch.save(model.state_dict(), savemodel_path)
        loaded_model=CFTSAD_Model(input_size=num_features,hidden_size=64,num_heads=4,num_layers=2,output_size=num_features).to(device)
# loaded_model=Transformer(input_size=num_features,hidden_size=64, num_heads=4)
        loaded_model.load_state_dict(torch.load(savemodel_path,map_location=device))
        loaded_model=loaded_model.to(device)
        anomaly_scores=anomaly_detection(loaded_model,test_dataloader,num_features,device)

        pre_res = evaluate(labels=label_test, scores=anomaly_scores, step=3000, adj=True)
        eval_res = eval_ad_result(pre_res, label_test)

    elif task_type == 'without_fre':
        model = Model_without_Fre(input_size=num_features, hidden_size=64, num_layers=2,output_size=num_features).to(device)
        print("task_type=without_fre")
        optimezer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()
        model = model.to(device)

        num_epochs = args.epochs
        for epoch in range(num_epochs):
            train_loss = train_model_without_Fre(model, train_dataloader, optimezer, device)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss}")
        #
        savemodel_path = f'D:/st_zp/CFTSAD/results/{args.dataset}/model_states/model_{args.dataset}_6_nofre.pth'
        torch.save(model.state_dict(), savemodel_path)
        loaded_model =Model_without_Fre(input_size=num_features, hidden_size=64,num_layers=2,output_size=num_features).to(device)
        loaded_model.load_state_dict(torch.load(savemodel_path, map_location=device))
        loaded_model = loaded_model.to(device)
        anomaly_scores = anomaly_detection_without_Fre(loaded_model, test_dataloader, num_features, device)

        pre_res = evaluate(labels=label_test, scores=anomaly_scores, step=300,adj=True)
        eval_res = eval_ad_result(pre_res, label_test)

    elif task_type == 'without_spa':
        model = Model_without_Spa(input_size=num_features, hidden_size=64, num_heads=4, num_layers=2,output_size=num_features).to(device)
        print("task_type=without_spa")
        optimezer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()
        model = model.to(device)

        num_epochs = args.epochs
        for epoch in range(num_epochs):
            train_loss = train_model(model, train_dataloader, optimezer, device)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss}")

        savemodel_path = f'D:/st_zp/CFTSAD/results/{args.dataset}/model_states/model_{args.dataset}_10.pth'
        # # save_path = 'D:/st_zp/Time-space correlation/results/model_states/model1.pkl'
        torch.save(model.state_dict(), savemodel_path)
        loaded_model =Model_without_Spa (input_size=num_features, hidden_size=64, num_heads=4, num_layers=2,output_size=num_features).to(device)
        # loaded_model=Transformer(input_size=num_features,hidden_size=64, num_heads=4)
        loaded_model.load_state_dict(torch.load(savemodel_path, map_location=device))
        loaded_model = loaded_model.to(device)
        anomaly_scores = anomaly_detection(loaded_model, test_dataloader, num_features, device)

        pre_res = evaluate(labels=label_test, scores=anomaly_scores, step=3000, adj=True)
        eval_res = eval_ad_result(pre_res, label_test)

    elif task_type == 'without_comb':
        model = Model_without_Comb(input_size=num_features, hidden_size=64, num_heads=4, num_layers=2,output_size=num_features).to(device)
        print("task_type=without_comb")
        optimezer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()
        model = model.to(device)

        num_epochs = args.epochs
        for epoch in range(num_epochs):
            train_loss = train_model(model, train_dataloader, optimezer, device)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss}")

        savemodel_path = f'D:/st_zp/CFTSAD/results/{args.dataset}/model_states/model_{args.dataset}_11.pth'
        # # save_path = 'D:/st_zp/Time-space correlation/results/model_states/model1.pkl'
        torch.save(model.state_dict(), savemodel_path)
        loaded_model =Model_without_Comb (input_size=num_features, hidden_size=64, num_heads=4, num_layers=2,output_size=num_features).to(device)
        # loaded_model=Transformer(input_size=num_features,hidden_size=64, num_heads=4)
        loaded_model.load_state_dict(torch.load(savemodel_path, map_location=device))
        loaded_model = loaded_model.to(device)
        anomaly_scores = anomaly_detection(loaded_model, test_dataloader, num_features, device)

        pre_res = evaluate(labels=label_test, scores=anomaly_scores, step=3000, adj=True)
        eval_res = eval_ad_result(pre_res, label_test)

    print(f"dataset_{args.dataset}_result")
    print("f1,precise,recall")
    print(eval_res)
    save_outpath=f'D:/st_zp/CFTSAD/results/{args.dataset}/out/{args.dataset}_result21.pkl'
    pkl_save(save_outpath,eval_res)
    print("result is saved")
