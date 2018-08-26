make clean
make -j5
out="output"
out+=$1
echo ${out}

echo $1 > input
echo "100" >> input 
./out/main < input > ${out}
for i in {0..50}
do
    echo $1 > input
    echo $((1000 + $i*2000)) >> input
    ./out/main < input >> ${out}
done
