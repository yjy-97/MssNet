#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import re
import datetime


def write_nii_addr(root_path, save_file, last_root_path):

    files = os.listdir(root_path)
    for file_name in files:
        next_root_path = os.path.join(root_path, file_name)
        if os.path.isdir(next_root_path):
            last_root_path = root_path
            root_path = next_root_path
            write_nii_addr(root_path, save_file, last_root_path)
            selected_path = select_slice_path(root_path)
            if (selected_path != "NONE"):
                save_file.writelines(selected_path + "\n")

            root_path = last_root_path

def select_slice_path(file_path):
    satisified_path = ['XSlice', 'YSlice', 'ZSlice']
    target_file = "NONE"
    for item in satisified_path:
        if item in file_path:
            print("file_path = {}".format(file_path))
            target_file = file_path
            break

    return target_file

def execute(root_path, save_file_name):
    save_file_path = os.path.join(root_path, save_file_name)
    if os.path.exists(save_file_path):
        i = datetime.datetime.now()
        date = str(i.year) + str(i.month) + str(i.day) + str(i.hour) + str(i.minute) + str(i.second)
        new_name = save_file_path + ".bak" + date
        os.rename(save_file_path, new_name)
        print("copied and deleted file, new_name = {}".format(new_name))
    # os.remove(save_file_path)

    with open(save_file_path, "a") as save_file:
        write_nii_addr(root_path, save_file, "")
    print("DONE... root_path = {}".format(root_path))



if __name__ == "__main__":

    root_path = r'F:\S_NC\gray_matter'
    save_file_name = 'S_NC_Slices_path.txt'
    execute(root_path, save_file_name)