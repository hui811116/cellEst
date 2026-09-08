"""
This script is used to batch process the TIFF files in a folder, merge the channels
using ImageJ macro, and save the merged images to a new folder.
Usage:
python script_pyimageJ_batch_chs.py <load_path> <save_path> <channel_type>
"""

import os
import sys
from pathlib import Path
import argparse
import imagej

parser = argparse.ArgumentParser()
parser.add_argument("load_path", help="Path to the folder containing the TIFF files")
parser.add_argument("save_path", help="Path to where the merged TIFF files will be saved")
parser.add_argument("channel_type",
                    choices=['nu','cy'],
                    help="Type of channel to merge (nu for nucleus, " \
                    "cy for cytoskeleton)")
parser.add_argument("--recursive", action="store_true", default=False, 
                    help="Whether to search for TIFF files recursively in subfolders")

# initialize ImageJ2
# GLOBAL MACRO
ij = imagej.init('sc.fiji:fiji')

MACRO_NU = """
#@ String tiff_blue_path
#@ String tiff_blue
#@ String save_name

open(tiff_blue_path);
run("Merge Channels...", " c3="+tiff_blue+" keep");
saveAs("tiff",save_name);
close("*");
"""
MACRO_CY = """
#@ String tiff_green_path
#@ String tiff_green
#@ String save_name

open(tiff_green_path);
run("Merge Channels...", "c2="+tiff_green+" keep");
saveAs("tiff",save_name);
close("*");
"""
def merge_channels(c_path,n_path,s_name,channel_type):
    """merge the channels using imageJ macro, save the merged image 
    to s_name, channel_type is either "nu" or "cy" """
    # name is the .tiff only?

    cytoskeleton_path = os.path.split(c_path)[-1]
    nucleus_path = os.path.split(n_path)[-1]
    saved_name = s_name
    tiff_file = nucleus_path
    tf_f2 = cytoskeleton_path
    if channel_type=="nu":
        ij_args = {
                "tiff_blue_path":n_path,
                "tiff_blue":tiff_file,
                "save_name":saved_name,
                }
        macro = MACRO_NU
    elif channel_type=="cy":
        ij_args = {
                "tiff_green_path":c_path,
                "tiff_green":tf_f2,
                "save_name":saved_name,
                }
        macro = MACRO_CY
    else:
        sys.exit()
    # run the imageJ macro to merge channels
    ij.py.run_macro(macro,ij_args)

def pair_tiffs(tiff_folder,recursive=False):
    """pair the tiff files in the folder, return a list of dict with 
    keys: id, ch1, ch2, ch3 (if exist)
    Matching by the path, except the channel information,
    which is the part of the filename, e.g. r01c02f03p01-ch1sk1fk1fl1.tiff, 
    where ch1 is the channel information,    
    """
    tiff_dict = {}
    for fpath in Path(tiff_folder).rglob("*.tiff") if recursive \
        else Path(tiff_folder).glob("*.tiff"):
        path_split = os.path.split(fpath)
        folders = path_split[:-1]
        fname = path_split[-1]
        fname_list = fname.split("-")
        # r01c02f03p01-ch1sk1fk1fl1.tiff
        # match by the path except the channel information, which is the part of the filename, 
        # e.g. r01c02f03p01-ch1sk1fk1fl1.tiff, where ch1 is the channel information
        # the path except the channel information is r01c02f03p01, which is the first part of 
        # the filename before the channel information, which is the part of the filename after
        img_id = fname_list[0]
        path_key = os.path.join(*folders,fname_list[0])
        if not path_key in tiff_dict:
            tiff_dict[path_key] = {"path":path_key, "id":img_id}
        ch_tex = fname_list[1][:3]
        if ch_tex == "ch1":
            tiff_dict[path_key]['ch1'] = fpath
        elif ch_tex == "ch2":
            tiff_dict[path_key]['ch2'] = fpath
        elif ch_tex == "ch3":
            tiff_dict[path_key]['ch3'] = fpath
        else:
            raise RuntimeError(f"undefined channel format {ch_tex}")
    tiff_list = []
    for _,v in tiff_dict.items():
        if "ch1" in v.keys() and "ch2" in v.keys() and "path" in v.keys():
            tiff_list.append(v)
    return tiff_list

def main(arg):
    """main function to process the tiff files and merge channels"""
    load_path = arg.load_path
    save_path = arg.save_path # base save path
    chtype = arg.channel_type
    if chtype not in ["nu", "cy"]:
        print("undefined channel type, abort")
        sys.exit()
    # get tiff filenames
    try:
        tiff_pairs = pair_tiffs(load_path, recursive=arg.recursive)
    except FileNotFoundError as e:
        print(f"Error in pairing TIFF files: {e}")
        sys.exit()
    os.makedirs(save_path,exist_ok=True)

    for item in tiff_pairs:
        img_path = item['path'] # this include the load_path
        # ch1(nucleus) --> blue
        # ch2(cytoskeleton) --> green
        n_path = item['ch1']
        c_path = item['ch2']
        # the id include the folder,
        # we want to keep the structure but change the first folder to the merge_path, 
        # and change the filename to the id with .tiff suffix
        # Example: load_folder/r01c02f03p01-ch1sk1fk1fl1.tiff --> save_folder/r01c02f03p01.tiff
        # replace the load_path with merge_path, only the first occurrence
        path_replace_load = img_path.replace(load_path, save_path, 1)
        save_full_path = f"{path_replace_load}_type_{chtype}.tiff"
        # make sure the save folder exists, because of the recursive structure
        save_folder = os.path.split(save_full_path)[0]
        os.makedirs(save_folder, exist_ok=True)
        merge_channels(c_path,n_path,save_full_path,chtype)

    print(f"Saving {len(tiff_pairs)} composite .tiff to:{save_path}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
