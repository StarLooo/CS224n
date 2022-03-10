import torch
from torchvision.models.resnet import *
import optparse
import datetime

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-g', '--logmsg', action="store", dest="logmsg", help="root directory",
                      default="Recursion-pytorch")
    options, args = parser.parse_args()
    # Print info about environments
    # logger = get_logger(options.logmsg, 'INFO') # noqa
    print('Cuda set up : time {}'.format(datetime.datetime.now().time()))
    device = torch.device('cuda')
    print('Device : {}'.format(torch.cuda.get_device_name(0)))
    print('Cuda available : {}'.format(torch.cuda.is_available()))
    n_gpu = torch.cuda.device_count()
    print('Cuda n_gpus : {}'.format(n_gpu))
