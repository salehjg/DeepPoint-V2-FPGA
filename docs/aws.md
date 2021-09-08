# Notes
* It is recommended to choose `r5.2xLarge` as the instance type. This would provide you with 64GB of RAM and 8 CPU-cores.
* If an instance type with plenty of CPU-cores and a small amount of RAM is selected, make sure to override `CpuCount` in the main `CMakeLists.txt` to prevent launching too many jobs at the same time which will cause Vivado to fatally crash with `failed to allocate memory` error.
* Do not use instance-specific ephemeral storage. This will lead to data loss (build directory) following a shutdown command issued by the automated build script. The instance types like `r5` instead of `r5d` do not offer ephemeral storage and they usually are cheaper.
* Choose appropriate AWS FPGA AMI version for your Deeppoint branch of choice. For SDx2019.1(branch `axi512`) the AMI version 1.7.0 should be selected and for Vitis2019.2(branch `vitis_axi512`) the AMI version 1.8.1 should be selected(or the lastest one).
* After terminating an instance, do not forget to manually delete detached EBS drives. Each FPGA AMI almost takes up 120GBs of SSD gp2 storage.

# Procedure
* Build the project on a powerful compute instance with FPGA AMI.
* Copy the build directory from the build server into a free tier t2.micro server.
* Convert the `*.xclbin` file into `*.awsxclbin`
* Zip the whole modified build directory. 
* Copy the zip file into the F1.2xLarge instance and run the host program with the first argument pointing to the `*.awsxclbin` file's path.

# General Server Setup(Any instance type with FPGA Development AMI)
```
# DO NOT use ephemeral storage(loses data after shutdown)
# Add extra storage of 20GBs of gp2 SSD for building the project
# https://objectivefs.com/howto/how-to-mount-instance-store-on-aws-ec2-instance-for-disk-cache

# Find the device name (normally nvme2n1 for FPGA AMI)
lsblk
sudo mkfs.ext4 -E nodiscard /dev/nvme2n1
sudo mkdir /mnt/mydrive
sudo mount /dev/nvme2n1 /mnt/mydrive
sudo chown -R $USER /mnt/mydrive
cd /mnt/mydrive
mkdir 00_workspace
cd 00_workspace
sudo yum -y install centos-release-scl epel-release nano cmake3 zip htop
sudo yum-config-manager --enable rhel-server-rhscl-7-rpms
sudo yum -y install devtoolset-7
#sudo pip install PasteBin #only for old commits of this repo
#sudo amazon-linux-extras install python3 #for the new pastebin script(1v1) and the bank_opt script. # python36 is already installed
sudo pip3 install requests
git clone https://github.com/aws/aws-fpga.git $AWS_FPGA_REPO_DIR
source /home/centos/src/project_data/aws-fpga/sdaccel_setup.sh
echo 'export AWS_PLATFORM=/home/centos/src/project_data/aws-fpga/SDAccel/aws_platform/xilinx_aws-vu9p-f1-04261818_dynamic_5_0/xilinx_aws-vu9p-f1-04261818_dynamic_5_0.xpfm' >> ~/.bashrc
#echo 'source /opt/xilinx/xrt/setup.sh' >> ~/.bashrc
source ~/.bashrc




```
# Cloning Deeppoint
```
cd /mnt/mydrive/00_workspace
git clone --recursive https://gitlab.com/salehjg/deeppoint-v1-fpga.git
cd deeppoint-v1-fpga
git checkout origin/axi512
git checkout axi512
git submodule update --init --recursive
mkdir build
cd build
```
# Building on AWS
```
# Check out config/CMakeLists.txt
# Enter username, pass and APIKEY for pastebin and DO NOT FORGET to turn on the pastebin agent(set to "1")

# use screen command to run the building script in the background in case of ssh disconnection.
# This command opens up a new subterminal
screen -S buildrun01 

scl enable devtoolset-7 bash
cmake3 ..

# Run the build script in this subterminal
sh autobuild_hw.sh

# Now press ctrl+A, then press D to send the subterminal to background

# Later, u can check to see if it's finished or not
screen -ls
screen -r buildrun01

# To terminate the subterminal from inside of the subterminal
# Press ctrl+D
```

# Converting XclBin to AWSXclBin
First copy the build folder from your build server into the free tier server with FPGA AMI, then:
1. Configure secret keys:
```
$ aws configure
AWS Access Key ID [None]: CSV Access key we just downloaded 
AWS Secret Access Key [None]: CSV Access Key we just downloaded 
Default region name [None]: us-east-1
Default output format [None]: json
```

2. Convert the xclbin file to awsxclbin and upload it to S3:
```
source $AWS_FPGA_REPO_DIR/sdaccel_setup.sh
cd into directory of your *.xclbin
sh $AWS_FPGA_REPO_DIR/SDAccel/tools/create_sdaccel_afi.sh -xclbin=<YOUR *.xclbin FNAME ONLY WITH EXTENSION> -s3_bucket=myfpgabucket -s3_dcp_key=DCP -s3_logs_key=Log
```
You must get `Successfully completed '/opt/xilinx/xrt/bin/xclbincat'` at the end of the script's std_out.

3. Wait until the process is finished:
First copy `FpgaImageId` from the output of the following command:
```
cat *afi_id.txt
```
Then:
```
aws ec2 describe-fpga-images --fpga-image-ids <FpgaImageId>
```
When the procedure is finished, you will not see `"Code": "pending"` in the output anymore. The process usually takes an hour or two.

# Running the project on FPGA(AWS)
1. Build the host side programs
```
make -j8
# this creates HostTest and DeepPointV1FPGA and a bunch of other CpuTests in build/test
```
2. Run CpuTests to make sure everything is set
```
make test
```
3. Copy the *.awsxclbin generated from the previous steps this the current build directory. Note that the *.xclbin is not needed and only the *.awsxclbin is enough.
4. Run HostTest that just tests the real FPGA device with some data transfers on the used DDR banks. (without any kernel launches)
```
sudo sh
source /opt/xilinx/xrt/setup.sh
./HostTest DeepPointV1FPGA_hw.awsxclbin
```
5. Run the project
```
sudo sh
source /opt/xilinx/xrt/setup.sh
sh LaunchDeepPointV1FPGA.sh #select mode 3
# this runs the project using selected modelarch and logs the host side program's output in DeepPointV1FPGA_Host_Log.txt
```
