import argparse, os, pickle
from utils import repair_dfn

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_dir', type=str, required=True,
                            help='Path to the directory containing the object data, each object should be a single .dat file.')
    args = arg_parser.parse_args()

    in_dir = args.data_dir
    fn_list = sorted([f for f in os.listdir(in_dir) if f.endswith(".dat")])

    for fn in fn_list:
        print(f"Repairing {fn}")
        fn_path = os.path.join(in_dir, fn)
        obj = pickle.load(open(fn_path, "rb"))
        for part in obj:
            breakpoint()


        # repair_dfn(obj)

        # # backup the original file
        # os.rename(fn_path, fn_path + ".bak")
        # pickle.dump(obj, open(fn_path, "wb"))
        