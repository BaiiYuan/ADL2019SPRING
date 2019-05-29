import torch

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def add_arguments(parser):
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('-msp', '--model_saved_path', type=str, default='./model', help="Model Saved Path")
    parser.add_argument('-tl', '--testing_labels', type=str, default=None, help="Testing Label Offered By TA")
    parser.add_argument('-od', '--output_dir', type=str, default=None, help="Output Directory")

    # parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    # parser.add_argument('-dp', '--data_path', type=str, default='../data', help="Data Path")

    return parser
