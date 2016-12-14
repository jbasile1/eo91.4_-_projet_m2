var=0
for i in $(seq 1 1 100);
do

echo $i $(./svm-train -s 0 -c 5 -g 0.55 -t 2 -e $i -v 10  -q    train_libsvm) >> experience_3
done
