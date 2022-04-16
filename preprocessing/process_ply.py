# ---------- Copyright (C) 2022 Qing Feng ----------
# 3d dataset preprocessing for 3dgan dataset
# voxelize and convert ply files to mat files

# Example: python3 process_ply.py 64 ../dataset/plys

import os
import sys
import open3d as o3d
import numpy as np
import scipy.io as io

def main():
    args = sys.argv[1:]
    cube_len = int(args[0])
    dir = args[1]
    odir = dir + "_mat"
    if not os.path.exists(odir):
        os.system("mkdir " + odir)

    try:
        # loop over .ply files in the dir
        for ply_file in os.listdir(dir):
            # skip non-obj files
            if not ply_file.endswith(".ply"):
                continue
            path = dir + "/" + ply_file

            # read .ply and write .mat file to the output
            pcd = o3d.io.read_point_cloud(path)

            # fit to unit cube
            pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
                    center=pcd.get_center())

            # voxelize
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1/(cube_len-1))

            # translate voxels to 3darray
            data = np.zeros((cube_len,cube_len,cube_len), int)

            for voxel in voxel_grid.get_voxels():
                x, y, z = voxel.grid_index
                data[x,y,z] = 1

            # rotate x
            data = np.rot90(data,1,(1,2))
            
            # get bounding box and offset
            maxx = maxy = maxz = 0
            minx = miny = minz = cube_len-1

            for x in range(cube_len):
                for y in range(cube_len):
                    for z in range(cube_len):
                        if data[x,y,z]:
                            maxx = max(x, maxx); maxy = max(y, maxy); maxz = max(z, maxz)
                            minx = min(x, minx); miny = min(y, miny); minz = min(z, minz)
            offsetx = cube_len//2-((maxx+minx+1)//2)
            offsety = cube_len//2-((maxy+miny+1)//2)
            offsetz = cube_len//2-((maxz+minz+1)//2)

            # recenter
            out = np.zeros((cube_len,cube_len,cube_len), int)
            for x in range(cube_len):
                for y in range(cube_len):
                    for z in range(cube_len):
                        if data[x,y,z]:
                            out[x+offsetx,y+offsety,z+offsetz] = 1

            # save file
            io.savemat(odir + "/" + ply_file[:-3] + "mat", {'instance': out})
            print(f"---------- Processed {ply_file} ----------")
    except:
        print("!!! Processing failed !!!")
    finally:
        print("---------- Preprocessing finished ----------\n")

if __name__ == "__main__":
    main()