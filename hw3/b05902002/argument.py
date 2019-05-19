def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--hidden', type=int, default=256, help='hidden size')
    parser.add_argument('-dr', '--drop_p', type=float, default=0.3)
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    return parser
