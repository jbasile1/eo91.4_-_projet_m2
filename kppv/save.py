def save(predictions, testing_set, training_set, k, accuracy, accuracy_by_class):
    """
    """
    f = None
    tested_date = datetime.datetime.now()

    tested_date = str(tested_date).split('.')
    tested_date = tested_date[0].replace(' ', '_')
    tested_date = (str(tested_date) + '_K' + str(k) + '_App' +
                   str(len(training_set)) + '_Gen' + str(len(testing_set)))

    print 'Results saved as', tested_date + '.csv'
    path = './results/'
    if not os.path.isdir(path):
        os.mkdir(path)

    # Saving the results of the predictions
    with open(path + tested_date + '.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter = ',',
                                quotechar = '|', quoting = csv.QUOTE_MINIMAL)

        # Adding the CSV header
        spamwriter.writerow(['predicted', 'actual'])

        # Adding the results
        for i in range(len(testing_set)):
            spamwriter.writerow([predictions[i], testing_set[i][-1]])

        # Saving the tests characteristics
    with open(path + tested_date + '.txt', 'w') as f:
        f.write ('Train sets size          : ' + repr(len(training_set)))
        f.write ('\nTest sets size           : ' + repr(len(testing_set)))
        f.write ('\nK                        : ' + str(k))

        # Print the accuracy for each class
        accuracy_by_class = get_accuracy_by_class(testing_set, predictions)
        f.write ('\n'*2 + 'Classes found in the set : ' + str(accuracy_by_class[0]))
        f.write ('\nSize of the set          : ' + str(accuracy_by_class[1]))

        f.write('\n'*2 + 'Accuracy        : ' + repr(accuracy) + '%')

        f.write ('\n'*2 + 'Accuracy per class')
        for pc in accuracy_by_class[2].keys():
            f.write( '\n\tClass' + str(pc) + '(' +repr(accuracy_by_class[2][pc])+
                     ')' + ':' + repr(accuracy_by_class[3][pc]) + '%')
