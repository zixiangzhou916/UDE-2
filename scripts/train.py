from __future__ import absolute_import, division, print_function
import argparse, os, sys, yaml, importlib
sys.path.append(os.getcwd())
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=str, default="ude")
    parser.add_argument('--task', type=str, default='UDETrainer',  
                        help='task, choose from [VQTokenizerTrainer, AutoEncoderTrainer, UDETrainer]')
    parser.add_argument('--config', type=str, default='configs/ude/config_ude_exp6.yaml', help='path to the config file')
    parser.add_argument('--dataname', type=str, default='HumanML3D', help='name of dataset, choose from [AMASS, AMASS-single, HumanML3D')
    parser.add_argument('--training_folder', type=str, default='logs/ude/', help='path of training folder')
    parser.add_argument('--training_name', type=str, default='exp6', help='name of the training')
    parser.add_argument('--a2m_trg_start_index', type=int, default=8, help='number of primitive motion tokens for audio-to-motion task')
    parser.add_argument('--training_mode', type=str, default="t2m", help='mode of training')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    Agent = importlib.import_module(r".{:s}.trainer".format(args.module), package="modules").__getattribute__(args.task)(args, config)
    Agent.train()
