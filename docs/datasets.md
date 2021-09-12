# Datasets
The data directory of repository is excluded from the git repository in order to separate 
the binary files from the code history. The compressed archive of the `data` directory could be found
at [the repository's release page](https://github.com/salehjg/DeepPoint-V2-FPGA/releases/download/data/data.zip). Please unzip the archive into `data/`.

As discussed in our paper, two datasets are used to evaluate our project. 
[The model](https://github.com/WangYueFt/dgcnn/tree/master/tensorflow) is trained using Tensorflow 1.x on 
an Nvidia GTX1070. To accelerate the training procedure on `ShapeNet V2 Core`, the trained weights 
for `ModelNet40` have been used as the initial values.

## ModelNet40
This dataset contains 40 CAD object classes and the samples for evaluation are stored as separate data and label numpy files (`*.npy`).

## ShapeNet V2 Core
Unlike ModelNet40, ShapeNet is consisted of 55 CAD objects offered in Mesh format (`*obj`). 
The mesh to point cloud conversion for `ShapeNet V2 Core` is done using our [opensource utility](https://github.com/salehjg/MeshToPointcloudFPS) and its [python script](https://github.com/salehjg/Shapenet2_Preparation).


