import imagej
import os
import sys
import pathlib
import numpy as np
import pandas as pd
import argparse

# initialize ImageJ2
# load local test data
ij = imagej.init('sc.fiji:fiji')

parser = argparse.ArgumentParser()
parser.add_argument("image_path",type=str,help="path/to/image/folder")
parser.add_argument("save_path",type=str,help="path/to/save/merged/images")
parser.add_argument("--recursive",action="store_true",default=False,help="search .tiff files recursively")
args = parser.parse_args()

"""
The script will keep the folder structure as the original image folder
but only save the merged images
"""

IMAGEJ_MACRO = """
#@ String tiff_green_path
#@ String tiff_blue_path
#@ String tiff_green
#@ String tiff_blue
#@ String save_name

open(tiff_blue_path);
open(tiff_green_path);
run("Merge Channels...", "c2="+tiff_green+" c3="+tiff_blue+" keep");
saveAs("tiff",save_name);
close("*");
"""
def mergeChannels(c_path,n_path,s_name):
    # name is the .tiff only?

    cytoskeleton_path = os.path.split(c_path)[-1]
    nucleus_path = os.path.split(n_path)[-1]
    saved_name = s_name
    tiff_file = nucleus_path
    tf_f2 = cytoskeleton_path
    args = {
        "tiff_green_path":c_path,
        "tiff_green":tf_f2,
        "tiff_blue_path":n_path,
        "tiff_blue":tiff_file,
        "save_name":saved_name,
    }
    result = ij.py.run_macro(IMAGEJ_MACRO,args)

def pairTiffs(tiff_folder,recursive=False):
    tiff_dict = {}
    folder = pathlib.Path(tiff_folder)
    fpaths = folder.rglob("*.tiff") if recursive else folder.glob("*.tiff")
    for fpath in fpaths:
        print(f"[LOG]{fpath}")
        fname_list = fpath.name.split("-")
        # r01c02f03p01-ch1sk1fk1fl1.tiff
        screenshot_id = fname_list[0]
        if not screenshot_id in tiff_dict.keys():
            tiff_dict[screenshot_id] = {"id":screenshot_id}
        ch_tex = fname_list[1][:3]
        if ch_tex == "ch1":
            tiff_dict[screenshot_id]['ch1'] = str(fpath)
        elif ch_tex == "ch2":
            tiff_dict[screenshot_id]['ch2'] = str(fpath)
        elif ch_tex == "ch3":
            tiff_dict[screenshot_id]['ch3'] = str(fpath)
        else:
            raise RuntimeError("undefinded channel format {:}".format(ch_tex))
    sys.exit()
    tiff_list = []
    for k,v in tiff_dict.items():
        #print(v)
        #if len(v) == 4: # id, ch1, ch2, ch3
        #    tiff_list.append(v)
        if "ch1" in v.keys() and "ch2" in v.keys() and "id" in v.keys():
            tiff_list.append(v)
    return tiff_list

def main(args):
    """main function to merge the channels of the tiff files in the folder"""
    load_path = args.image_path
    save_path = args.save_path
    folder_name = os.path.split(load_path)[-1]
    merge_path = os.path.join(save_path,folder_name) 
    

    # get tiff filenames
    tiff_pairs = pairTiffs(load_path,recursive=args.recursive)

    if len(tiff_pairs) == 0:
        print("No tiff pairs found in {:}".format(load_path))
        return sys.exit(0)
    os.makedirs(merge_path,exist_ok=True)
    for item in tiff_pairs:
        id = item['id']
        # ch1(nucleus) --> blue
        # ch2(cytoskeleton) --> green
        n_path = item['ch1']
        c_path = item['ch2']
        save_to = os.path.join(merge_path,"{:}_merge".format(id))
        mergeChannels(c_path,n_path,save_to)

    print("Saving {:} composite .tiff to:{:}".format(len(tiff_pairs),merge_path))

if __name__ == "__main__":
    main(args)