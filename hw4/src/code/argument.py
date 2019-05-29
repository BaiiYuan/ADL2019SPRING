import torch

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def add_arguments(parser):
    # parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    # parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    # parser.add_argument('-dp', '--data_path', type=str, default='../data', help="Data Path")
    return parser
