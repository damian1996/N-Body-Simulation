#make clean
if [ $1 == "debug" ]
then
    make
    gdb out/main
elif [ $1 == "test" ]
then
    make
    for i in {1..10}
    do
        ./out/main < input
    done
elif [ $1 == "prof" ]
then
    make
    /usr/local/cuda-9.2/bin/nvprof ./out/main < input
else
    make
    ./out/main < input
fi


/usr/local/cuda-9.2/bin/nvprof ./out/main
