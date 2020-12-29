import argparse
import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch
from datasets import ImageDataset


def test(Unet):
    if Unet:
        from Models import Generator_resnet_unet as Generator
    else:
        from Models import Generator_resnet as Generator
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='data/horse2zebra/', help='root directory of the dataset')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--generator_A2B', type=str, default='output/netG_AtoB.pth', help='A2B generator checkpoint file')
    opt = parser.parse_args()
    print(opt)


    netG_A2B = Generator(opt.input_nc, opt.output_nc)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("you are using", device)
    netG_A2B.to(device)

    # Load state dicts
    netG_A2B.load_state_dict(torch.load(opt.generator_A2B))

    # Set model's test mode
    netG_A2B.eval()

    # Dataset loader
    transforms_ = [transforms.Resize((256 * 2, 256 * 2)),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'),
                            batch_size=opt.batchSize, shuffle=False, num_workers=0)

    # Create output dirs if they don't exist
    if not os.path.exists('output/X2Y'):
        os.makedirs('output/X2Y')

    print('generating fake images by generator......')
    for i, batch in enumerate(dataloader):

        # Real images
        real_A = batch['A'].to(device)
        Real_A = 0.5 * (real_A + 1.0)

        # Generate fake output
        temp1 = netG_A2B(real_A).data
        fake_B = 0.5 * (temp1 + 1.0)

        AB = torch.cat([Real_A, fake_B])


        # Save image files
        save_image(AB, 'output/X2Y/%04d.png' % (i + 1))
    print('test done')