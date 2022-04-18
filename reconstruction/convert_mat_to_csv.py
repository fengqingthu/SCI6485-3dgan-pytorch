import os
import sys
import scipy.io as io
import numpy as np

def main():
    args = sys.argv[1:]
    cube_len = int(args[0])
    vsize = 1/cube_len*200

    dir = args[1]
    odir = dir + "_csv"
    if not os.path.exists(odir):
        os.system("mkdir " + odir)
    
    try:
        # loop over .ply files in the dir
        for mat_file in os.listdir(dir):
            # skip non-obj files
            if not mat_file.endswith(".mat"):
                continue
            path = dir + "/" + mat_file
            voxels = io.loadmat(path)['instance']
            data = []
            index = 0
            for x in range(cube_len):
                for y in range(cube_len):
                    for z in range(cube_len):
                        if voxels[x,y,z]:
                            data.append((index, x*vsize, y*vsize, z*vsize))
                            index += 1
            
            data = np.array(data)
            np.savetxt(odir + "/" + mat_file[:-3] + "csv", data, fmt = '%.15f')
            print(f"---------- Processed {mat_file} ----------")
    
    except:
        print("!!! Processing failed !!!")
    finally:
        print("---------- Preprocessing finished ----------\n")
            
if __name__ == "__main__":
    main()