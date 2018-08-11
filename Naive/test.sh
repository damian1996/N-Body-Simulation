make clean
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
else
    make
    ./out/main
fi