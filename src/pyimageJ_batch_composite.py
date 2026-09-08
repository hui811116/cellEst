import imagej
import os
import sys
import glob
import numpy as np
import pandas as pd

# initialize ImageJ2
# load local test data
ij = imagej.init('sc.fiji:fiji')

argv = sys.argv
macro = """
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
    result = ij.py.run_macro(macro,args)

def pairTiffs(tiff_folder):
    tiff_dict = {}
    for fpath in glob.glob(os.path.join(tiff_folder,"*.tiff")):
        #print(f"[LOG]{fpath}")
        fname_list = os.path.split(fpath)[-1].split("-")
        # r01c02f03p01-ch1sk1fk1fl1.tiff
        screenshot_id = fname_list[0]
        if not screenshot_id in tiff_dict.keys():
            tiff_dict[screenshot_id] = {"id":screenshot_id}
        ch_tex = fname_list[1][:3]
        if ch_tex == "ch1":
            tiff_dict[screenshot_id]['ch1'] = fpath
        elif ch_tex == "ch2":
            tiff_dict[screenshot_id]['ch2'] = fpath
        elif ch_tex == "ch3":
            tiff_dict[screenshot_id]['ch3'] = fpath
        else:
            raise RuntimeError("undefinded channel format {:}".format(ch_tex))
    tiff_list = []
    for k,v in tiff_dict.items():
        #print(v)
        #if len(v) == 4: # id, ch1, ch2, ch3
        #    tiff_list.append(v)
        if "ch1" in v.keys() and "ch2" in v.keys() and "id" in v.keys():
            tiff_list.append(v)
    return tiff_list

load_path = argv[1]
save_path = argv[2]
#print(os.path.split(load_path))
folder_name = os.path.split(load_path)[-1]
#print(folder_name)
#sys.exit()
merge_path = os.path.join(save_path,folder_name) 
os.makedirs(merge_path,exist_ok=True)

# get tiff filenames
tiff_pairs = pairTiffs(load_path)

for item in tiff_pairs:
    id = item['id']
    # ch1(nucleus) --> blue
    # ch2(cytoskeleton) --> green
    n_path = item['ch1']
    c_path = item['ch2']
    save_to = os.path.join(merge_path,"{:}_merge".format(id))
    #print(n_path,c_path)
    #print(save_to)
    #sys.exit()
    mergeChannels(c_path,n_path,save_to)

print("Saving {:} composite .tiff to:{:}".format(len(tiff_pairs),merge_path))
