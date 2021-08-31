cd ../cmake-build-debug
cd test/ocltests/
echo $PWD
emconfigutil --platform ${AWS_PLATFORM} --nd 1
export XCL_EMULATION_MODE=hw_emu

