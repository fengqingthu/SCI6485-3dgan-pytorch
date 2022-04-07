# ---------- Copyright (C) 2022 Qing Feng ----------
# 3d dataset preprocessing for 3dgan dataset
# voxelize and convert obj files to mat files
# For MacOs 10.10 above, please install XQuatz
# Example: python3 process_obj.py 64 ../dataset/A_NormalClothes

import os
import sys
import binvox_rw
import scipy.io as io

def main():
    args = sys.argv[1:]
    cube_len = args[0]
    dir = args[1]
    odir = dir + "_mat"
    if not os.path.exists(odir):
        os.system("mkdir " + odir)

    try:
        # loop over .obj files in the dir
        for obj_file in os.listdir(dir):
            # skip non-obj files
            if not obj_file.endswith(".obj"):
                continue
            # voxelize it to .binvox, write intermediate file to the same dir
            path = dir + "/" + obj_file
            if not os.path.exists(path[:-3] + "binvox"):
                os.system("./binvox -e -cb -rotx -d " + str(cube_len) + " " + path)
            # read .binvox and write .mat file to the output
            with open(path[:-3] + "binvox","rb") as f:
                data = binvox_rw.read_as_3d_array(f).data.astype(int)
                io.savemat(odir + "/" + obj_file[:-3] + "mat", {'instance': data})
            # clean intermediate .binvox files
            os.system("rm " + path[:-3] + "binvox")
    finally:
        # clean any intermediate files
        os.system("rm " + dir + "/*.binvox")

if __name__ == "__main__":
    main()