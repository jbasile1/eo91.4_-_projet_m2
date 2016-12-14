var=0
for i in $(seq 0 0.05 10);
do

echo $i $(./svm-train -s 0 -c 5 -g $i -t 2 -e 0.5 -v 10  -q    train_libsvm) >> experience2
done
