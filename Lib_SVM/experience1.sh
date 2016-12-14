
for i in `seq 0 30`
do
echo $i $(./svm-train -s 0 -c 5 -g 0.5 -t 2 -e 0.5 -v 10  -q  -d $i  train_libsvm) >> experience1
done

