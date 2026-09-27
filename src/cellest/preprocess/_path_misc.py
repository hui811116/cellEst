import pathlib
import os


def pair_tiffs_chs(tiff_folder,recursive=False):
    """pair the tiff files in the folder, return a list of dict with 
    keys: id, ch1, ch2, ch3 (if exist)
    Matching by the path, except the channel information,
    which is the part of the filename, e.g. r01c02f03p01-ch1sk1fk1fl1.tiff, 
    where ch1 is the channel information,    
    """
    tiff_dict = {}
    for fpath in pathlib.Path(tiff_folder).rglob("*.tiff") if recursive \
        else pathlib.Path(tiff_folder).glob("*.tiff"):
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

# def pair_tiffs(tiff_folder,recursive=False):
#     tiff_dict = {}
#     folder = pathlib.Path(tiff_folder)
#     fpaths = folder.rglob("*.tiff") if recursive else folder.glob("*.tiff")
#     for fpath in fpaths:
#         fname_list = fpath.name.split("-")
#         # r01c02f03p01-ch1sk1fk1fl1.tiff
#         screenshot_id = fname_list[0]
#         if not screenshot_id in tiff_dict.keys():
#             tiff_dict[screenshot_id] = {"id":screenshot_id}
#         ch_tex = fname_list[1][:3]
#         if ch_tex == "ch1":
#             tiff_dict[screenshot_id]['ch1'] = str(fpath)
#         elif ch_tex == "ch2":
#             tiff_dict[screenshot_id]['ch2'] = str(fpath)
#         elif ch_tex == "ch3":
#             tiff_dict[screenshot_id]['ch3'] = str(fpath)
#         else:
#             raise RuntimeError("undefinded channel format {:}".format(ch_tex))
#     tiff_list = []
#     for k,v in tiff_dict.items():
#         if "ch1" in v.keys() and "ch2" in v.keys() and "id" in v.keys():
#             tiff_list.append(v)
#     return tiff_list