



import os
import argparse
from glob import glob
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# --------------------------- Dataset ---------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, image_size=64):
        self.paths = []
        for ext in ('*.png','*.jpg','*.jpeg'):
            self.paths.extend(glob(os.path.join(root_dir, '**', ext), recursive=True))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img)

# --------------------------- Networks (DCGAN-style) ---------------------------

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

# --------------------------- Training Loop ---------------------------

def train(args):
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')
    print(f"Using device: {device}")

    # dataset
    dataset = ImageFolderDataset(args.data_dir, image_size=args.image_size)
    print(f"Found {len(dataset)} images")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # models
    netG = Generator(nz=args.nz, ngf=args.ngf, nc=3).to(device)
    netD = Discriminator(nc=3, ndf=args.ndf).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(args.sample_size, args.nz, 1, 1, device=device)

    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    iters = 0

    os.makedirs(args.output_dir, exist_ok=True)
    samples_dir = os.path.join(args.output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, data in enumerate(pbar):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            netD.zero_grad()
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if iters % args.log_interval == 0:
                pbar.set_postfix({'errD': errD.item(), 'errG': errG.item(), 'D(x)': D_x, 'D(G(z))': D_G_z2})

            if iters % args.sample_interval == 0:
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                utils.save_image((fake + 1) / 2, os.path.join(samples_dir, f'sample_{iters:06d}.png'), nrow=8)

            iters += 1

        # end epoch
        # save checkpoints
        torch.save({'epoch': epoch+1, 'netG_state': netG.state_dict(), 'netD_state': netD.state_dict(), 'optimizerG': optimizerG.state_dict(), 'optimizerD': optimizerD.state_dict()},
                   os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    # final save
    torch.save(netG.state_dict(), os.path.join(args.output_dir, 'generator_final.pth'))
    torch.save(netD.state_dict(), os.path.join(args.output_dir, 'discriminator_final.pth'))

    print('Training finished. Models and samples saved to', args.output_dir)

# --------------------------- Utilities ---------------------------

def sample(args):
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')
    netG = Generator(nz=args.nz, ngf=args.ngf, nc=3).to(device)
    netG.load_state_dict(torch.load(args.gen_path, map_location=device))
    netG.eval()

    with torch.no_grad():
        noise = torch.randn(args.sample_size, args.nz, 1, 1, device=device)
        fake = netG(noise).detach().cpu()
        out_path = os.path.join(args.output_dir, 'samples_manual')
        os.makedirs(out_path, exist_ok=True)
        utils.save_image((fake + 1) / 2, os.path.join(out_path, 'sample_manual.png'), nrow=8)
    print('Saved samples to', out_path)

# --------------------------- CLI ---------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to folder with images')
    parser.add_argument('--output_dir', type=str, default='./output', help='Where to save samples and checkpoints')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--nz', type=int, default=100, help='Size of z latent vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--sample_size', type=int, default=64)
    parser.add_argument('--sample_interval', type=int, default=500)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--mode', type=str, default='train', choices=['train','sample'])
    parser.add_argument('--gen_path', type=str, default='', help='Path to saved generator (for sampling)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'train':
        train(args)
    else:
        if not args.gen_path:
            raise ValueError('Provide --gen_path when using sample mode')
        sample(args)
