import argparse
from Generate_target_result import run_target
from MIA_attack import run_MIA
from Calculate_PLD import run_PLD

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-rep', type=int, nargs='+', default=5, help='number of repeating experimants')
    parser.add_argument('--file_list', type=int, nargs='+', default=[0, 1, 2], help='which dataset')
    args = parser.parse_args()

    run_target(args)
    run_MIA(args)
    run_PLD(args)