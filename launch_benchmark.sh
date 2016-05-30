for i in $(seq 1 36)
do
echo $i
python benchmark.py $i 2>/dev/null
done
