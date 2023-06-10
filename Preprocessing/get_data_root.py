#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import re
import time
import datetime


def write_nii_addr(root_path, file_path, original_doc, gray_matter_doc, white_matter_doc, CSF_doc, lable):
    files = os.listdir(file_path)
    for file_name in files:
        _file_path = os.path.join(file_path, file_name)
        if os.path.isdir(_file_path):
            write_nii_addr(root_path, _file_path, original_doc, gray_matter_doc, white_matter_doc, CSF_doc, lable)
        else:
            postfix = file_name.split('.')[1]
            if (postfix == "nii"):
                pre_fix = file_name.split('.')[0]
                if (re.match('mwp1', pre_fix)):
                    _name = lable + "_gray_matter.txt"
                    with open(os.path.join(root_path, _name), "a") as f:
                        f.writelines(_file_path + "\n")

                elif (re.match('mwp2', pre_fix)):
                    _name = lable + "_white_matter.txt"
                    with open(os.path.join(root_path, _name), "a") as f:
                        f.writelines(_file_path + "\n")

                elif (re.match('wm', pre_fix)):
                    _name = lable + "_CSF.txt"
                    with open(os.path.join(root_path, _name), "a") as f:
                        f.writelines(_file_path + "\n")

                else:
                    _name = lable + "_original.txt"
                    with open(os.path.join(root_path, _name), "a") as f:
                        f.writelines(_file_path + "\n")

            # print(os.path.join(file_path))


def create_modal_file(root_path, lable):
    original_doc = os.path.join(root_path, "original")
    gray_matter_doc = os.path.join(root_path, "gray_matter")
    white_matter_doc = os.path.join(root_path, "white_matter")
    CSF_doc = os.path.join(root_path, "CSF")

    if not os.path.exists(original_doc):
        print("Create file original_doc = {}".format(original_doc))
        os.makedirs(original_doc)

    if not os.path.exists(gray_matter_doc):
        print("Create file gray_matter_doc = {}".format(gray_matter_doc))
        os.makedirs(gray_matter_doc)

    if not os.path.exists(white_matter_doc):
        print("Create file white_matter_doc = {}".format(white_matter_doc))
        os.makedirs(white_matter_doc)

    if not os.path.exists(CSF_doc):
        print("Create file CSF_doc = {}".format(CSF_doc))
        os.makedirs(CSF_doc)

    import shutil
    backup_file = os.path.join(root_path, "backup")
    i = datetime.datetime.now()
    date = str(i.year) + str(i.month) + str(i.day)
    if not os.path.exists(backup_file):
        print("Create file backup_file = {}".format(backup_file))
        os.makedirs(backup_file)
    files = os.listdir(root_path)
    for file in files:
        print("[backup] file = {}".format(file))
        if not os.path.isdir(file):
            if (len(file.split('.')) > 1):
                if (file.split('.')[1] == "txt"):
                    old_name = file
                    new_name = date + "_" + str(file)
                    print("old_name = {}".format(old_name))
                    print("new_name = {}".format(new_name))
                    os.rename(os.path.join(root_path, old_name), os.path.join(root_path, new_name))
                    source_dir = os.path.join(root_path, new_name)
                    target_dir = os.path.join(root_path, "backup")
                    shutil.copy(source_dir, target_dir)
                    os.remove(source_dir)

    file_path = root_path
    write_nii_addr(root_path, file_path, original_doc, gray_matter_doc, white_matter_doc, CSF_doc, lable)


if __name__ == "__main__":
    root_path_AD = r'F:\S_NC'
    create_modal_file(root_path_AD, "S_NC")

