import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision

def train_DCGAN(dataloader, generator, discriminator, criterion, 
                optimizerG, optimizerD, device, num_epochs=5, nz = 100):
    # 결과를 저장할 리스트
    G_losses = []
    D_losses = []
    img_list = []
    
    generator = generator.to(device);
    discriminator = discriminator.to(device)

    # 훈련 시작
    print(">>> POONGSAN: Start training ..")

    for epoch in range(num_epochs):
        D_avg_loss = 0; G_avg_loss = 0;
        for i, data in enumerate(dataloader, 0):
            
            # ===== 판별자(Discriminator) 훈련 =====
            discriminator.zero_grad()
            
            # 실제 데이터
            real_data = data.to(device)
            real_labels = torch.full((real_data.size(0),), 1.0, device=device) # 실제 데이터는 1로 레이블
            output = discriminator(real_data).view(-1)
            errD_real = criterion(output, real_labels)
            errD_real.backward()
            
            # 가짜 데이터
            noise = torch.randn(real_data.size(0), nz, 1, 1, device=device)
            fake_data = generator(noise)
            
            fake_labels = torch.full((real_data.size(0),), 0.0, device=device) # 가짜 데이터는 0으로 레이블
            output = discriminator(fake_data.detach()).view(-1)
            errD_fake = criterion(output, fake_labels)
            errD_fake.backward()
            
            errD = errD_real + errD_fake
            optimizerD.step()
            
            # ===== 생성자(Generator) 훈련 =====
            generator.train()
            generator.zero_grad()
            output = discriminator(fake_data).view(-1)
            errG = criterion(output, real_labels) # 생성자는 판별자가 가짜를 실제로 판별하길 원함
            errG.backward()
            optimizerG.step()

            D_avg_loss += errG.item();
            G_avg_loss += errD.item();
        
        G_losses.append(G_avg_loss / (i + 1))
        D_losses.append(D_avg_loss / (i + 1))

        print("EPOCH{0} | G_loss {1} D_loss {2}"
                .format(epoch, G_avg_loss / (i + 1), D_avg_loss / (i + 1)))
        
        if epoch % 10 == 0:  # 예: 100 에포크마다 저장
            generator.eval()
            with torch.no_grad():
                noise = torch.randn(32, nz, 1, 1, device=device)
                fake_data = generator(noise).detach().cpu()
            img_list.append(fake_data)
            grid = np.transpose(torchvision.utils.make_grid(fake_data, padding=2, normalize=True), (1, 2, 0))
            plt.imshow(grid)
            plt.savefig(f'result/epoch_{epoch}.png')
    
    # 로스함수 시각화 
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('result/loss_graph.png')

    # 실험 로그 저장
    with open('result/experiment_log.txt', 'w') as f:
        f.write("Epoch\tGenerator Loss\tDiscriminator Loss\n")
        for epoch, (g_loss, d_loss) in enumerate(zip(G_losses, D_losses)):
            f.write(f"{epoch}\t{g_loss}\t{d_loss}\n")

    
    return G_losses, D_losses
