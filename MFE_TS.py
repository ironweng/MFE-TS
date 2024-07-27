import torch
import torch.nn as nn
from base_models import FeaturesCombine,TransformerEncoder,TransformerDecoder,SeqReconstruction,Complex_spa_temLayer,TransformerAutoencoder


class Fre_domainAutoencoder(nn.Module):
    def __init__(self, input_size_fre, hidden_size_fre, num_heads):
        super(Fre_domainAutoencoder, self).__init__()
        self.combined_model = FeaturesCombine(input_size_fre, hidden_size_fre)
        self.transformer_autoencoder = TransformerAutoencoder(input_size_fre, hidden_size_fre, num_heads)
        self.reconstruction_model = SeqReconstruction(input_size_fre, hidden_size_fre)

    def forward(self, amp, phase):
        combined_output = self.combined_model(amp, phase)
        reconstructed_output = self.transformer_autoencoder(combined_output)
        reconstructed_seq = self.reconstruction_model(reconstructed_output)
        return reconstructed_seq

class Fre_domainAutoencoder_without_Comb(nn.Module):
    def __init__(self, input_size_fre, hidden_size_fre, num_heads):
        super(Fre_domainAutoencoder_without_Comb, self).__init__()
        self.combined_model = FeaturesCombine(input_size_fre, hidden_size_fre)
        self.transformer_autoencoder = TransformerAutoencoder(input_size_fre, hidden_size_fre, num_heads)
        self.reconstruction_model = SeqReconstruction(input_size_fre, hidden_size_fre)

    def forward(self, amp, phase):
        combined_output = (amp+phase)/2
        reconstructed_output = self.transformer_autoencoder(combined_output)
        reconstructed_seq = self.reconstruction_model(reconstructed_output)
        return reconstructed_seq

class Tem_domainAutoencoder(nn.Module):
    def __init__(self, input_size_tem, hidden_size_tem, num_layers, output_size_tem):
        super(Tem_domainAutoencoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size_tem, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=input_size_tem, out_channels=hidden_size_tem, kernel_size=2,padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=hidden_size_tem, kernel_size=3,padding=2)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=hidden_size_tem, kernel_size=5)
        self.relu = nn.ReLU
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size_tem, num_layers=num_layers, batch_first=True)  #用于捕捉时间序列中的长期依赖关系
        self.attention=nn.Linear(hidden_size_tem,1)    #注意力层，将LSTM的输出进行加权平均，其中权重由softmax函数计算得出，通过这种方式，模型可以学到不同属性在时间序列中的重要性，进而捕捉到空间相关性
        self.maxpool=Complex_spa_temLayer(hidden_size_tem,hidden_size_tem)
        self.deconv1=nn.ConvTranspose1d(in_channels=hidden_size_tem, out_channels=hidden_size_tem, kernel_size=2,stride=2)
        self.deconv2 = nn.ConvTranspose1d(in_channels=hidden_size_tem, out_channels=32, kernel_size=3)   #逆卷积层，用于将时间——空间特征转化回时间序列
        self.proj = nn.Sequential(
                nn.ReLU(),
                nn.Linear(32,hidden_size_tem),
                nn.ReLU(),
                nn.Linear(hidden_size_tem,output_size_tem)
             )
        self.fc=nn.Linear(32, output_size_tem)
            #最后经过全连接层输出重构的多变量时间序列

    def forward(self, x,masked_ratio):
        mask_tensor = torch.ones_like(x)
        mask_ratio = masked_ratio
        total_elements = x.numel()
        masked_elements = int(total_elements * mask_ratio)
        random_indices = torch.randperm(total_elements)[:masked_elements]
        mask_tensor.view(-1)[random_indices] = 0
        x = x * mask_tensor
        x_ori = x.transpose(1, 2)
        # print("x_ori",x.shape)
        x_s = self.conv1(x_ori)
        x_s = x_s.transpose(1, 2)
        x_s, _ = self.lstm(x_s)
        attention_weight=torch.softmax(self.attention(x_s),dim=1)
        R_s=torch.mul(x_s,attention_weight)
        R_s=self.maxpool(R_s)
        # print("R_s",R_s.shape)
        x_t=self.conv2(x_ori)
        # print('x_t',x_t.shape)
        x_t=self.conv3(x_t)
        # print('x_t',x_t.shape)
        x_t = self.conv4(x_t)
        # print('x_t', x_t.shape)
        # x_t = self.relu(self.conv2(x_ori))
        # x_t = self.relu(self.conv3(x_t))
        # x_t = self.relu(self.conv4(x_t))
        R_t=self.maxpool(x_t.transpose(1, 2))
        # print("R_t", R_t.shape)
        R=torch.cat((R_t,R_s),dim=2)
        # print("R",R.shape)
        # R = self.deconv1(R)
        R=self.deconv2(R)
        # x=self.deconv2(x)
        R = R.transpose(1, 2)
        x = self.proj(R)
        # x=self.fc(x)
        # print('x_pre',x.shape)
        return x

class Tem_domainAutoencode_without_spa(nn.Module):
    def __init__(self, input_size_tem, hidden_size_tem, num_layers, output_size_tem):
        super(Tem_domainAutoencode_without_spa, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size_tem, out_channels=32, kernel_size=3)  # 用于提取时间相关特征
        self.conv2 = nn.Conv1d(in_channels=input_size_tem, out_channels=hidden_size_tem, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=hidden_size_tem, kernel_size=3)
        self.relu = nn.ReLU
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size_tem, num_layers=num_layers,
                            batch_first=True)  # 用于捕捉时间序列中的长期依赖关系
        self.attention = nn.Linear(hidden_size_tem,
                                   1)  # 注意力层，讲LSTM的输出进行加权平均，其中权重由softmax函数计算得出，通过这种方式，模型可以学到不同属性在时间序列中的重要性，进而捕捉到空间相关性
        self.maxpool = Complex_spa_temLayer(hidden_size_tem, hidden_size_tem)  #
        self.deconv1 = nn.ConvTranspose1d(in_channels=hidden_size_tem, out_channels=hidden_size_tem, kernel_size=2,
                                          stride=2)
        self.deconv2 = nn.ConvTranspose1d(in_channels=hidden_size_tem, out_channels=32,
                                          kernel_size=3)  # 逆卷积层，用于将时间——空间特征转化回时间序列
        self.proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32, hidden_size_tem),
            nn.ReLU(),
            nn.Linear(hidden_size_tem, output_size_tem)
        )
        self.fc = nn.Linear(32, output_size_tem)
        # 最后经过全连接层输出重构的多变量时间序列

    def forward(self, x, masked_ratio):
        mask_tensor = torch.ones_like(x)
        mask_ratio = masked_ratio
        total_elements = x.numel()
        masked_elements = int(total_elements * mask_ratio)
        random_indices = torch.randperm(total_elements)[:masked_elements]
        mask_tensor.view(-1)[random_indices] = 0
        x = x * mask_tensor
        x_ori = x.transpose(1, 2)
        x_t = self.relu(self.conv2(x_ori))
        x_t = self.relu(self.conv3(x_t))
        R_t = self.maxpool(x_t.transpose(1, 2))
        R = self.deconv2(R_t)
        # x=self.deconv2(x)
        R = R.transpose(1, 2)
        x = self.proj(R)
        # x=self.fc(x)
        # print('x_pre',x.shape)
        return x

class CFTSAD_Model(nn.Module):
    def __init__(self,input_size,hidden_size,num_heads,num_layers,output_size ):
        super(CFTSAD_Model, self).__init__()
        self.fre_encoder = Fre_domainAutoencoder(input_size_fre=input_size,
                                                 hidden_size_fre=hidden_size,
                                                 num_heads=num_heads
                                                 )
        self.tem_encoder =Tem_domainAutoencoder(input_size_tem=input_size,
                                                 hidden_size_tem=hidden_size,
                                                 num_layers=num_layers,
                                                 output_size_tem=output_size
                                                )

    # def criterion(self):
    #     criterion_loss = nn.MSELoss()
    #     return criterion_loss

    def forward(self,t,m,p,masked_ratio):
        # original_t=t
        reconstructed_t1=self.tem_encoder(t,masked_ratio)
        reconstructed_t2=self.fre_encoder(m,p)
        # loss_tem = self.criterion(reconstructed_t1,original_t)
        # loss_fre=self.criterion(reconstructed_t2,original_t)

        return reconstructed_t1,reconstructed_t2


class Model_without_Fre(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Model_without_Fre, self).__init__()
        self.tem_encoder = Tem_domainAutoencoder(input_size_tem=input_size,
                                                 hidden_size_tem=hidden_size,
                                                 num_layers=num_layers,
                                                 output_size_tem=output_size
                                                 )
    def forward(self,t,masked_ratio):
        # original_t=t
        reconstructed_t=self.tem_encoder(t,masked_ratio)
        # loss_tem = self.criterion(reconstructed_t1,original_t)
        # loss_fre=self.criterion(reconstructed_t2,original_t)

        return reconstructed_t

class Model_without_Spa(nn.Module):
    def __init__(self,input_size,hidden_size,num_heads,num_layers,output_size ):
        super(Model_without_Spa, self).__init__()
        self.fre_encoder = Fre_domainAutoencoder(input_size_fre=input_size,
                                                 hidden_size_fre=hidden_size,
                                                 num_heads=num_heads
                                                 )
        self.tem_encoder =Tem_domainAutoencode_without_spa(input_size_tem=input_size,
                                                 hidden_size_tem=hidden_size,
                                                 num_layers=num_layers,
                                                 output_size_tem=output_size
                                                )

    # def criterion(self):
    #     criterion_loss = nn.MSELoss()
    #     return criterion_loss


    def forward(self,t,m,p,masked_ratio):
        # original_t=t
        reconstructed_t1=self.tem_encoder(t,masked_ratio)
        reconstructed_t2=self.fre_encoder(m,p)
        # loss_tem = self.criterion(reconstructed_t1,original_t)
        # loss_fre=self.criterion(reconstructed_t2,original_t)

        return reconstructed_t1,reconstructed_t2

class Model_without_Comb(nn.Module):
    def __init__(self,input_size,hidden_size,num_heads,num_layers,output_size ):
        super(Model_without_Comb, self).__init__()
        self.fre_encoder = Fre_domainAutoencoder_without_Comb(input_size_fre=input_size,
                                                 hidden_size_fre=hidden_size,
                                                 num_heads=num_heads
                                                 )
        self.tem_encoder =Tem_domainAutoencoder(input_size_tem=input_size,
                                                 hidden_size_tem=hidden_size,
                                                 num_layers=num_layers,
                                                 output_size_tem=output_size
                                                )

    # def criterion(self):
    #     criterion_loss = nn.MSELoss()
    #     return criterion_loss


    def forward(self,t,m,p,masked_ratio):
        # original_t=t
        reconstructed_t1=self.tem_encoder(t,masked_ratio)
        reconstructed_t2=self.fre_encoder(m,p)
        # loss_tem = self.criterion(reconstructed_t1,original_t)
        # loss_fre=self.criterion(reconstructed_t2,original_t)

        return reconstructed_t1,reconstructed_t2
