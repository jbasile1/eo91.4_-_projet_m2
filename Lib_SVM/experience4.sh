var=0
for i in $(seq 1 1 1000);
do

echo $i $(./svm-train -s 0 -c $i -g 0.05 -t 2 -e 1 -v 10  -q    train_libsvm_brute) >>experience_4_brute
done
