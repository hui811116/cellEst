"""
This script is used to batch process the TIFF files in a folder, merge the channels
using ImageJ macro, and save the merged images to a new folder.
Usage:
python script_pyimageJ_batch_chs.py <load_path> <save_path> <channel_type>
"""

import os
import sys
import argparse
import tqdm
from cellest.preprocess import pair_tiffs_chs, imagej_init, imagej_run_macro, imagej_macro_select, merge_type_alias

parser = argparse.ArgumentParser()
parser.add_argument("load_path", help="Path to the folder containing the TIFF files")
parser.add_argument("save_path", help="Path to where the merged TIFF files will be saved")
parser.add_argument("channel_type",
                    choices=['nu','cy','composite'],
                    help="Type of channel to merge (nu for nucleus, " \
                    "cy for cytoskeleton, composite for both)")
parser.add_argument("--recursive", action="store_true", default=False, 
                    help="Whether to search for TIFF files recursively in subfolders")

def main(arg):
    """main function to process the tiff files and merge channels"""
    load_path = arg.load_path
    save_path = arg.save_path # base save path
    chtype = arg.channel_type
    if chtype not in ["nu", "cy", "composite"]:
        print("undefined channel type, abort")
        sys.exit()
    print("[INFO] selected channel type: {:}".format(merge_type_alias(chtype)))
    # get tiff filenames
    try:
        tiff_pairs = pair_tiffs_chs(load_path, recursive=arg.recursive)
    except FileNotFoundError as e:
        print(f"Error in pairing TIFF files: {e}")
        sys.exit()

    if len(tiff_pairs) == 0:
        print(f"No tiff pairs found in {load_path}")
        return sys.exit(0)
    os.makedirs(save_path,exist_ok=True)

    # initialize ImageJ2
    ij = imagej_init()

    for item in tqdm.tqdm(tiff_pairs):
        img_path = item['path'] # this include the load_path
        # ch1(nucleus) --> blue
        # ch2(cytoskeleton) --> green
        n_path = item['ch1']
        c_path = item['ch2']
        # the id include the folder,
        # we want to keep the structure but change the first folder to the merge_path, 
        # and change the filename to the id with .tiff suffix
        # Example: load_folder/r01c02f03p01-ch1sk1fk1fl1.tiff --> save_folder/r01c02f03p01.tiff
        # replace the load_path with merge_path, keep the folder structure
        # 
        path_replace_load = img_path.replace(load_path, save_path, 1)
        save_full_path = f"{path_replace_load}_type_{merge_type_alias(chtype)}.tiff"
        # make sure the save folder exists, because of the recursive structure
        save_folder = os.path.split(save_full_path)[0]
        os.makedirs(save_folder, exist_ok=True)

        imj_macro, ij_args = imagej_macro_select(c_path,n_path,save_full_path,merge_type_alias(chtype))
        result = imagej_run_macro(ij,imj_macro,ij_args)


    print(f"Saving {len(tiff_pairs)} composite .tiff to:{save_path}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
