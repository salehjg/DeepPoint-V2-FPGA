# Run on tcl.post route(only for a HW build)
write_checkpoint -force ${PostRouteCheckpointFile}
report_power -hier all -file ${PowerReportTextFile} -rpx ${PowerReportRpxFile}
report_utilization -slr -file ${SlrUtilOutputFile}
report_utilization -pblocks pblock_pcie -pblocks pblock_SH -pblocks pblock_SH_SHIM -file ${PblockUtilOutputFile}
