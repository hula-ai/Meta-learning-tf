import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default="test", help='train or test')
parser.add_argument('--restore_training', default=False, help='restores the last trained model to resume training')
parser.add_argument('--label_type', default="one_hot", help='one_hot or five_hot')
parser.add_argument('--n_classes', default=5, help='number of classes to be selected (randomly) in each episode')
parser.add_argument('--seq_length', default=15, help='number of samples selected from all classes in each episode')
parser.add_argument('--sample_strategy', default='uniform', help='sampling strategy from classes; random or uniform')
parser.add_argument('--augment', default=True, help='decides to run the augmentation on data or not')
parser.add_argument('--max_angle', default=40, help='max random rotation angle when applying augmentation')
parser.add_argument('--max_shift', default=3, help='max pixel shift when applying augmentation')
parser.add_argument('--model', default="MANN", help='LSTM, MANN, or NTM')
parser.add_argument('--read_head_num', default=4, help='number of read heads')
parser.add_argument('--batch_size', default=16)
parser.add_argument('--num_episodes', default=100000, help='total number of training episodes')
parser.add_argument('--learning_rate', default=1e-3)
parser.add_argument('--rnn_size', default=200, help='number of hidden units of the LSTM controller')
parser.add_argument('--image_width', default=20)
parser.add_argument('--image_height', default=20)
parser.add_argument('--rnn_num_layers', default=1)
parser.add_argument('--memory_size', default=128, help='number of memory slots')
parser.add_argument('--memory_vector_dim', default=40, help='size of each memory slot')
parser.add_argument('--shift_range', default=1, help='only for model=NTM')
parser.add_argument('--write_head_num', default=1, help='only for model=NTM. For MANN #(write_head) = #(read_head)')
parser.add_argument('--test_batch_num', default=10, help='number of batches in test mode')
parser.add_argument('--n_train_classes', default=1100)
parser.add_argument('--n_test_classes', default=423)
parser.add_argument('--data_dir', default='../data/omniglot_resized', help='data directory')
parser.add_argument('--save_dir', default='./saved_model')
parser.add_argument('--save_freq', default=5000, help='save the trained model after this many episodes')
parser.add_argument('--save_hdf5', default=True, help='to save the train/test results in separate HDF5 files')
parser.add_argument('--disp_freq', default=100, help='display and save the train/test after this many episodes')
parser.add_argument('--test_freq', default=1000, help='run and display the test results after this many episodes')
parser.add_argument('--tensorboard_dir', default='./summary/')
parser.add_argument('--noise_strategy', default='random', help='random, or uniform')
parser.add_argument('--noise_size', default=0, help='portion of support samples to be noisy')
parser.add_argument('--reduce_spt_size', default=0.5, help='portion of reduced size of the support set')
args = parser.parse_args()
