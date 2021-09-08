# Debugging Host-side in CLion
In order to debug the host-side program in any modes(`sw_emu`, `hw_emu`, or `system`), CLion or any other C++ IDE could be used.

Remember to run `scripts/DbgScriptBinDir.sh` or `scripts/DbgScriptOclTestDir` before starting debugging session for the main executable or the OclTests' main executable. Note that class `XilinxImplementation` is configured to select `sw_emu` in the case that variable `XCL_EMULATION_MODE` was not set beforehand.  

# Launching Vivado HLS
It is possible to launch Vivado HLS GUI and optimize the kernel of choice. This could be done after running a `hw_emu` build:
```
cd _x/task_<KERNEL>_solution/task_<KERNEL>
vivado_hls -p task_<KERNEL>
```
* Please note that any changes to the source files will be reflected on the main repository files.
