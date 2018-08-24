#make clean
make -j5
if [ $1 == "debug" ]
then
    gdb out/main
elif [ $1 == "debugCuda" ]
then
    export CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1
    /usr/local/cuda/bin/cuda-gdb out/main
elif [ $1 == "FinTest" ]
then
    echo "4" > input
    echo "10" >> input 
    ./out/main < input
    for i in {1..10}
    do
        echo "4" > input
        echo $(($i*10000)) >> input
        ./out/main < input
    done
elif [ $1 == "test" ]
then
    for i in {1..10}
    do
        ./out/main < input
    done
elif [ $1 == "prof" ]
then
    /usr/local/cuda-9.2/bin/nvprof ./out/main < input
else
    ./out/main < input
fi

