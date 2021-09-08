# TCL scripts
There are two TCL scripts named `PreRoute.tcl` and `PostRoute.tcl`. 
- The pre-route script is reponsible for generating the placed design `Pre_route_checkpoint.dcp` file.
- The post-route script generates SLR and per-block utilization and power estimation reports along with the routed design `Post_route_checkpoint.dcp` file.

Please note that Vivado could be used to open the design checkpoint files (`*.dcp`) to further explore the design implementation space. Open Vivado and choose `File: Checkpoint: Open` menu to import the desired checkpoint file.

The power estimation reports are generated in two formats (`*.txt` and `*.rpx`). The interactive report file (`*.rpx`) could be opened using the following Vivado TCL command: `open_report -file <path to *.rpx> -name myreport`.
