# Memory Bank Optimizer
`scripts/bank_optimizer_v2.py` is created to find optimized memory banks for the kernels with the objective of minimizing the number of `DataMover` kernel launches needed to execute the computation graph of the selected `TopModelArch`.

# Instructions
1. Run the selected `CModel` on FPGA for `hostlog_0trace.log` to be created.
2. Open `hostlog_0trace.log` and find the lines that after `Dumping bank-crossing logs for` and append them.
3. Copy the these lines into `bank_optimizer_v2.py`, method `get_objective` and line `objective =`
4. Assign the allowed banks per kernel like `banks_transpose=[1,2]` to allow banks one and two to be selected for kernel `Transpose`, or `banks_transpose=[1]` to force the kernel to use only the bank one.
5. Run the script.
6. Use the output to configure `config` submodule of the main `DeepPoint-V2-FPGA` repository and then rebuild the FPGA image.
