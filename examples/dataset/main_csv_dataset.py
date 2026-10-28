import yaml
import sys
import argparse
import datetime
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    #parser.add_argument("output_csv", type=str, help="Path to the output CSV file.")
    parser.add_argument("--output_folder", type=str, default=".", help="Folder to save the output CSV file.")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    # use the .yaml file data card to create a .csv two-column data
    # the leading name is for the dataset name
    # if seeing field "classes", for each label, recursively find images 
    # the "paths" field in "labels" list all folders that need to be checked
    # put the relative path of images and the label as a two-column CSV file, 
    # the first column is the relative path of the image, the second column is the label
    
    # the default dataset should append the datetime if not specified, to avoid overwriting previous datasets
    dataset_name = config.get("name", "dataset")
    if not config.get("name"):
        dataset_name += f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    df = pd.DataFrame(columns=["uri", "class_name"])
    """
    name: experiment_natural_versus_chemical_senescent
    description: A dataset comparing natural and chemical senescent processes.
    classes:
        - label: natural_senescent
            description: Samples representing natural senescence processes.
            paths:
                - processed/adscs_take_3/P12
        - label: chemical_senescent
            description: Samples representing chemically induced senescence processes.
            paths:
                - doxtreated/senescent
        - label: control
            description: Control samples without senescence induction.
            paths:
                - processed/adscs_take_3/P4
                - doxtreated/control
    """
    cnt = 0
    for class_info in config.get("classes", []):
        # this is a list
        if not isinstance(class_info, dict):
            print(f"Invalid class_info format: {class_info}. Expected a dictionary.")
            continue
        if "label" in class_info:
            label_dict = class_info["label"]
            label_name = label_dict['name']
            cnt += 1
            for path in label_dict["paths"]:
                # Recursively find all image files in the specified path
                for image_path in Path(path).rglob("*"):
                    if image_path.is_file() and image_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
                        df = pd.concat([df, pd.DataFrame([[str(image_path), label_name]], columns=["uri", "class_name"])], ignore_index=True)
        else:
            # can support different data formats, such as "attributes"
            raise ValueError(f"Missing 'label' key in class_info: {class_info}")


    # summary of the dataset
    print(f"Dataset Name: {dataset_name}")
    print(f"Total Samples: {len(df)}")
    print("Class Distribution:")
    print(df["class_name"].value_counts())
    # Save the DataFrame to a CSV file
    output_csv = f"{dataset_name}.csv"
    df.to_csv(f"{args.output_folder}/{output_csv}", index=False)