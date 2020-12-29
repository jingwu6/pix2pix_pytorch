import argparse
import Train
import Test


def retrieve_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=80, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=3, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='data/horse2zebra/', help='root directory of the dataset')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=40,
                        help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    arg = parser.parse_args()
    print(arg)
    return arg







if __name__ == '__main__':
    arg = retrieve_args()
    modell = Train. PixelToPixel(Unet=False)
    dataloader = modell.dataloader(arg)
    Loss_Generator, Loss_Identity, Loss_GANs, x = modell.train(arg, dataloader)
    modell.PlotandSave(Loss_Generator, Loss_Identity, Loss_GANs, x)

    Test.test(Unet=False)