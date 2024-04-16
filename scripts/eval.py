import argparse, os, sys
sys.path.append(os.getcwd())
import importlib
import yaml
# os.environ['CURL_CA_BUNDLE'] = '' # SSLError: HTTPSConnectionPool(host='huggingface.co', port=443)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=str, default="ude", help="")
    parser.add_argument('--task', type=str, default='UDEEvaluator',  
                        help='task, choose from [VQTokenizerEvaluator, UDEEvaluator, ...]')
    parser.add_argument('--config', type=str, default='configs/ude/config_ude_exp5.yaml', help='path to the config file')
    parser.add_argument('--dataname', type=str, default='HumanML3D', help='name of dataset, choose from [AMASS, AMASS-single, HumanML3D')
    parser.add_argument('--eval_folder', type=str, default='logs/ude/eval/', help='path of training folder')
    parser.add_argument('--eval_name', type=str, default='exp5', help='name of the training choose from [t2m, a2m]')
    parser.add_argument('--topk', type=int, default=20, help='')
    parser.add_argument('--temperature', type=float, default=2.0, help='')
    parser.add_argument('--use_sas', type=str2bool, default=True, help="")
    parser.add_argument('--repeat_times', type=int, default='1', help='number of repeat times per caption')
    parser.add_argument('--eval_mode', type=str, default='t', help='evaluation mode, t means text-to-motion, a means audio-to-motion')
    parser.add_argument('--s2m_block_size', type=int, default=160, help="")
    parser.add_argument('--s2m_decode_long', type=str2bool, default=True, help="")
    parser.add_argument('--is_debug', type=str2bool, default=False, help='whether to to run debug mode')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    Agent = importlib.import_module(r".{:s}.evaluator".format(args.module), package="modules").__getattribute__(args.task)(args, config)
    Agent.eval()
    
