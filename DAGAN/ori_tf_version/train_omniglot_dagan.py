import argparse
import data as dataset
from experiment_builder import ExperimentBuilder

parser = argparse.ArgumentParser(description='Welcome to GAN-Shot-Learning script')
parser.add_argument('--batch_size', nargs="?", type=int, default=32, help='batch_size for experiment')
parser.add_argument('--discriminator_inner_layers', nargs="?", type=int, default=1,
                    help='discr_number_of_conv_per_layer')
parser.add_argument('--generator_inner_layers', nargs="?", type=int, default=1, help='discr_number_of_conv_per_layer')
parser.add_argument('--experiment_title', nargs="?", type=str, default="omniglot_dagan_experiment", help='Experiment name')
parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='continue from checkpoint of epoch')
parser.add_argument('--num_of_gpus', nargs="?", type=int, default=1, help='discr_number_of_conv_per_layer')
parser.add_argument('--gpus', nargs='?', type = str, default = '0', help = 'specific gpus')
parser.add_argument('--z_dim', nargs="?", type=int, default=100, help='The dimensionality of the z input')
parser.add_argument('--dropout_rate_value', type=float, default=0.5, help='dropout_rate_value')
parser.add_argument('--num_generations', nargs="?", type=int, default=64, help='num_generations')
parser.add_argument('--learning_rate', nargs = '?', type=float, default=1e-4, help = 'learning_rate')

args = parser.parse_args()
batch_size = args.batch_size
num_gpus = args.num_of_gpus
#set the data provider to use for the experiment
data = dataset.OmniglotDAGANDataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
                                    num_of_gpus=num_gpus, gen_batches=10)
#init experiment
experiment = ExperimentBuilder(parser, data=data)
#run experiment
experiment.run_experiment()
