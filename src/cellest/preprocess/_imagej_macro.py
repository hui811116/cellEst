import imagej
import os

def imagej_init():
    """initialize imageJ2"""
    ij = imagej.init('sc.fiji:fiji')
    return ij

def imagej_run_macro(ij,macro,args):
    """run imageJ macro"""
    result = ij.py.run_macro(macro,args)
    return result

def merge_type_alias(merge_type):
    """return alias dict for merge_type"""
    composite_alias = ["merge","composite"]
    nucleus_alias = ["nu","nucleus"]
    cytoskeleton_alias = ["cy","cytoskeleton"]
    alias_dict = {}
    for alias in composite_alias:
        alias_dict[alias] = "composite"
    for alias in nucleus_alias:
        alias_dict[alias] = "nucleus"
    for alias in cytoskeleton_alias:
        alias_dict[alias] = "cytoskeleton"
    if merge_type in alias_dict:
        return alias_dict[merge_type]
    else:
        raise ValueError(f"Invalid merge_type: {merge_type}")

def imagej_macro_select(c_path,n_path,s_name,merge_type):
    """select the macro to run based on the merge_type"""
    merge_type = merge_type_alias(merge_type)
    if merge_type=="composite":
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
        ij_args = {
            "tiff_green_path":c_path,
            "tiff_green":str(os.path.split(c_path)[-1]),
            "tiff_blue_path":n_path,
            "tiff_blue":str(os.path.split(n_path)[-1]),
            "save_name":s_name,
        }
    elif merge_type=="nucleus":
        macro = """
#@ String tiff_blue_path
#@ String tiff_blue
#@ String save_name

open(tiff_blue_path);
run("Merge Channels...", " c3="+tiff_blue+" keep");
saveAs("tiff",save_name);
close("*");
"""
        ij_args = {
            "tiff_blue_path":n_path,
            "tiff_blue":str(os.path.split(n_path)[-1]),
            "save_name":s_name,
        }
    elif merge_type=="cytoskeleton":
        macro = """
#@ String tiff_green_path
#@ String tiff_green
#@ String save_name

open(tiff_green_path);
run("Merge Channels...", "c2="+tiff_green+" keep");
saveAs("tiff",save_name);
close("*");
"""
        ij_args = {
            "tiff_green_path":c_path,
            "tiff_green":str(os.path.split(c_path)[-1]),
            "save_name":s_name,
        }
    else:
        raise ValueError(f"Invalid merge_type: {merge_type}")
    return macro, ij_args