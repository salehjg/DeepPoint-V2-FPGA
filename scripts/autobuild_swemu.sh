start=$SECONDS
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' = | tee autobuild_swemu_log.txt
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' = | tee -a autobuild_swemu_log.txt
make compile_swemu | tee -a autobuild_swemu_log.txt
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' = | tee -a autobuild_swemu_log.txt
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' = | tee -a autobuild_swemu_log.txt
make link_swemu | tee -a autobuild_swemu_log.txt
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' = | tee -a autobuild_swemu_log.txt
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' = | tee -a autobuild_swemu_log.txt

stop=$SECONDS
elapsed=$[stop-start]
hours=$(bc <<< "scale=3;${elapsed}/3600.0")
echo "Total Elapsed Seconds: ${elapsed}" | tee -a autobuild_swemu_log.txt
echo "Total Elapsed Hours: ${hours}" | tee -a autobuild_swemu_log.txt

RUN_PASTEBIN_OR_NOT=@PASTEBIN_0_1@
if [ "$RUN_PASTEBIN_OR_NOT" -eq 1 ] ; then
	echo "**Running the PasteBin script..."
	python3 PasteBinScript.py
	echo "**Done."
else
	echo "The PasteBin script is disabled."
fi

echo "DONE. RUNNING POWEROFF COMMAND..."
sudo poweroff
