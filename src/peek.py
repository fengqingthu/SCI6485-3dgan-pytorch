# from visdom import Visdom
import utils
import os
import params
import matplotlib.pyplot as plt

# method to plot one instance from the dataset using visdom
# def peekDatasetVis(path):
#     cfg = {"server": "localhost", "port": 8097}
#     vis = Visdom('http://' + cfg["server"], port = cfg["port"])
#     with open(path + os.listdir(path)[0], "rb") as f:
#         voxels = utils.getVoxelFromMat(f, params.cube_len)
#         utils.plotVoxelVisdom(voxels, vis, params.model_dir)

# method to plot one instance from the dataset
def peekDataset(path):
    for file in os.listdir(path):
        with open(path + file, "rb") as f:
            voxels = utils.getVoxelFromMat(f, params.cube_len)
            fig = plt.figure(figsize=(16,16))
            x, y, z = voxels.nonzero()
            ax = fig.gca(projection="3d")
            ax.axes.set_xlim3d(left=0.2, right=63.8) 
            ax.axes.set_ylim3d(bottom=0.2, top=63.8) 
            ax.axes.set_zlim3d(bottom=0.2, top=63.8) 
            ax.scatter(x, y, z, zdir='z', c='red')
            plt.savefig('../peek/{}.png'.format(file[:-4]), bbox_inches='tight')
            plt.close()
      
def main(path):
    os.system("mkdir ../peek")
    peekDataset(path)

if __name__ == "__main__":
    dataset_path = params.data_dir + params.model_dir + "64/train/"
    main(dataset_path)