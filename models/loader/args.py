
import argparse
import os
from configs.model_config import *


# Additional argparse types
def path(string):
    if not string:
        return ''
    s = os.path.expanduser(string)
    if not os.path.exists(s):
        raise argparse.ArgumentTypeError(f'No such file or directory: "{string}"')
    return s


def file_path(string):
    if not string:
        return ''
    s = os.path.expanduser(string)
    if not os.path.isfile(s):
        raise argparse.ArgumentTypeError(f'No such file: "{string}"')
    return s


def dir_path(string):
    if not string:
        return ''
    s = os.path.expanduser(string)
    if not os.path.isdir(s):
        raise argparse.ArgumentTypeError(f'No such directory: "{string}"')
    return s


parser = argparse.ArgumentParser(prog='psyDiagnose',
                                 description='About psyDiagnose, local knowledge based ChatGLM with langchain ｜ '
                                             '基于本地知识库的 ChatGLM 问答')

parser.add_argument('--no-remote-model', action='store_true', help='remote in the model on '
                                                                   'loader checkpoint, '
                                                                   'if your load local '
                                                                   'model to add the ` '
                                                                   '--no-remote-model`')
parser.add_argument('--model-name', type=str, default=LLM_MODEL, help='Name of the model to load by default.')
parser.add_argument("--use-lora",type=bool,default=USE_LORA,help="use lora or not")
parser.add_argument('--lora', type=str, default=LORA_NAME,help='Name of the LoRA to apply to the model by default.')
parser.add_argument("--lora-dir", type=str, default=LORA_DIR, help="Path to directory with all the loras")
parser.add_argument('--use-ptuning-v2',default=USE_PTUNING_V2,help="whether use ptuning-v2 checkpoint")
parser.add_argument("--ptuning-dir",type=str,default=PTUNING_DIR,help="the dir of ptuning-v2 checkpoint")
# Accelerate/transformers
parser.add_argument('--load-in-8bit', action='store_true', default=LOAD_IN_8BIT,
                    help='Load the model with 8-bit precision.')
parser.add_argument('--bf16', action='store_true', default=BF16,
                    help='Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.')

args = parser.parse_args([])
# Generares dict with a default value for each argument
DEFAULT_ARGS = vars(args)
