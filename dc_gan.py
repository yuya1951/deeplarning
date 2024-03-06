import os

import torch
from torch import nn, optim

from adabelief_pytorch import AdaBelief # AdaBelief を使う場合に使用

from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, datasets
import tqdm
from statistics import mean

from torch.utils.tensorboard import SummaryWriter
#%load_ext tensorboard

EPOCH = 20000        # エポック数
BATCH = 64          # バッチ数
LR = 0.0002           # 学習率       バージョン002では 0.0005 に設定
nz = 100                 # 潜在特徴100次元ベクトルz
SAVE_TIME = 100 # エポック数何回ごとに1度保存するか

# datasetの準備
dataset = datasets.ImageFolder("./bullpug/",
    transform=transforms.Compose([
        transforms.Resize((64, 64)),
        # transforms.CenterCrop(224), # 割愛
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) # 割愛
]))

# dataloaderの準備
data_loader = DataLoader(
    dataset,
    batch_size=BATCH,
    shuffle=True,
    num_workers=4 # 環境に合わせて要変更
)

# バージョン004のFRelu を使う場合
class FReLU(nn.Module):
    def __init__(self, in_c, k=3, s=1, p=1):
        super().__init__()
        self.f_cond = nn.Conv2d(in_c, in_c, kernel_size=k,stride=s, padding=p,groups=in_c)
        self.bn = nn.BatchNorm2d(in_c)

    def forward(self, x):
        tx = self.bn(self.f_cond(x))
        out = torch.max(x,tx)
        return out
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(

            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # FRelu(256), # バージョン004のFRelu を使う場合

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # FRelu(128), # バージョン004のFRelu を使う場合

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # FRelu(64), # バージョン004のFRelu を使う場合

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # FRelu(64), # バージョン004のFRelu を使う場合

            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(

            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.main(x).squeeze()
    
# 使用するGPU、またはCPUの指定
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE

model_G = Generator().to(DEVICE)
model_D = Discriminator().to(DEVICE)

params_G = optim.Adam(model_G.parameters(),
    lr=LR, # 学習率
    betas=(0.5, 0.999)
)

params_D = optim.Adam(
# params_D = AdaBelief( # パターン003 のAdaBelief を使う場合。パラメーター変更しないと実行できるもののWarningが表示される

    model_D.parameters(),
    lr=LR, # 学習率,
    betas=(0.5, 0.999)
)

# ロスを計算するときのラベル変数
ones = torch.ones(BATCH).to(DEVICE) # 正例 1
zeros = torch.zeros(BATCH).to(DEVICE)# 負例 0
loss_f = nn.BCEWithLogitsLoss()

# 途中結果の確認用の潜在特徴z
check_z = torch.randn(BATCH, nz, 1, 1).to(DEVICE)

# 訓練関数
def train_dcgan(model_G, model_D, params_G, params_D, data_loader):
    log_loss_G = []
    log_loss_D = []
    for real_img, _ in data_loader:
        batch_len = len(real_img)

        # == Generatorの訓練 ==
        # 偽画像を生成
        z = torch.randn(batch_len, nz, 1, 1).to(DEVICE)
        fake_img = model_G(z)

        # 偽画像の値を一時的に保存 => 注(１)
        fake_img_tensor = fake_img.detach()

        # 偽画像を実画像（ラベル１）と騙せるようにロスを計算
        out = model_D(fake_img)
        loss_G = loss_f(out, ones[: batch_len])
        log_loss_G.append(loss_G.item())

        # 微分計算・重み更新 => 注（２）
        model_D.zero_grad()
        model_G.zero_grad()
        loss_G.backward()
        params_G.step()


        # == Discriminatorの訓練 ==
        # sample_dataの実画像
        real_img = real_img.to(DEVICE)

        # 実画像を実画像（ラベル１）と識別できるようにロスを計算
        real_out = model_D(real_img)
        loss_D_real = loss_f(real_out, ones[: batch_len])

        # 計算省略 => 注（１）
        fake_img = fake_img_tensor

        # 偽画像を偽画像（ラベル０）と識別できるようにロスを計算
        fake_out = model_D(fake_img_tensor)
        loss_D_fake = loss_f(fake_out, zeros[: batch_len])

        # 実画像と偽画像のロスを合計
        loss_D = loss_D_real + loss_D_fake
        log_loss_D.append(loss_D.item())

        # 微分計算・重み更新 => 注（２）
        model_D.zero_grad()
        model_G.zero_grad()
        loss_D.backward()
        params_D.step()

    return mean(log_loss_G), mean(log_loss_D)

# TensorBoard用
writer = SummaryWriter()

# TensorBoard表示用
#%tensorboard --logdir=runs

for i in range(1, EPOCH+1):
    G_loss_mean, D_loss_mean = train_dcgan(model_G, model_D, params_G, params_D, data_loader)

    # TensorBoard D_Lossの書き出し
    writer.add_scalar('Discriminator',
                            G_loss_mean,
                            i
    )

    # TensorBoard G_Loss_Lossの書き出し
    writer.add_scalar('Generater',
                            G_loss_mean,
                            i
    )

    # 訓練途中のモデル・生成画像の保存
    if i % SAVE_TIME == 0:
        torch.save(
            model_G.state_dict(),
            f"./Weight_Generator5/G_{i}.pth",
            pickle_protocol=4)
        torch.save(
            model_D.state_dict(),
            f"./Weight_Discriminator5/D_{i}.pth",
            pickle_protocol=4)

        generated_img = model_G(check_z)
        save_image(generated_img,
                   f"./Generated_Image5/{i}.jpg")

    # 20回ごとにエポック数と処理の進捗割合を表示
    if i % 20 == 0:
        print(f"epoch:{i} progress:{round(i / EPOCH * 100 , 2)}%")

writer.close()
