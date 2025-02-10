# Bioemus - IBEA
Bioemus version compatible with Indicator-Based Evolutionary Algorithm (IBEA) to be installed on FPGA. 

**WARNING**: this code is different from the original one published [Nature Communication](https://doi.org/10.1038/s41467-024-48905-x), which is instead available on this [github url](https://github.com/Ceramic-Blue-Tim/bioemus/tree/main). It contains major changes in the configuration files (the "rat brain" model) and the possibility to stimulate from an output pin of the board (several changes in the code regarding bioemus-genoa). This repository can be cloned inside a new board and used to run locally the IBEA algorithm, directly on the Ubuntu onboard Processing System.

## Installation

### 1. Setup ubuntu 22.04 operating system for Kria
Follow the instructions at this [link](https://xilinx.github.io/kria-apps-docs/kr260/linux_boot/ubuntu_22_04/build/html/docs/intro.html) to install Ubuntu on the board (by flashing it in the microSD).
### 2. Connect via SSH
- Connect USB serial (1) to PC (USB micro B).
- Communicate with the board using Putty (Windows) or Minicom (macOS/Linux).
- After inserting the password, check the IP address with the `ifconfig` command (it should be under `eth0` or `eth1`, next the `inet:` word).
- Close the terminal and disconnect the USB cable.
- Connection via SSH `ssh ubuntu@[IPaddress]`
### 3. Setup the system to work with Bioemus
- Update system
```Bash
sudo apt-get update && sudo apt upgrade
sudo reboot now
```
> On Kria™ KR260 the boot time can be quite long (~180 seconds) from the TPM self test message.
- Install Xilinx Development Tools package
```Bash
# Ubuntu 22.04
sudo snap install xlnx-config --classic --channel=2.x
# Initialize xilinx config package
xlnx-config.sysinit
# Bootgen
sudo apt install bootgen-xlnx
```
- Install BioemuS dependencies
```Bash
# ZeroMQ
sudo apt install libzmq3-dev
# Python
sudo apt install python3-pip python3-pyqt5
```
- (optional, install Ubuntu Software)
```Bash
sudo snap install snap-store
```
- (optional, utilities)
```Bash
sudo apt install net-tools
sudo apt install devmem2
```
### 4. Clone this repository
Clone this repository on the board at `~/`. The folder should be renamed "bioemus".
```Bash
git clone https://github.com/barbaLab/bioemus-ibea.git bioemus
```
Build the drivers to work properly
```Bash
cd bioemus
source ./build.sh
```
## Usage

### Creating and launching a simulation
1. Configure the network using the `mainIBEA.ipynb` notebook. By running, it should generate a `.json` and a `.txt` files in the `config/` folder.
2. Call the bash script to begin the simulation:
```Bash
source init.sh
source ./launch_app ./config/[json_file.json]
```
This runs the simulation and produce a `.bin` file within the folder data, containing the raster plot of the simulation.
### Running IBEA algorithm
1. To run the IBEA algorithm directly within the ubuntu onboard system you have first to setup the python venv `IBEAvenv` by running on bash:
``` Bash
source setup_IBEA_venv.sh
source IBEAvenv/bin/activate
```
2. Configure and run the IBEA algorithm from the `mainIBEA` script. I suggest to open the folder with the Remote-SSH extension in vscode for better usability (from the host PC this extension will automatically install vscode on the board). Use the python kernel `IBEAvenv` to run the notebook.

