import math
from collections import Counter
from collections import defaultdict
from collections import OrderedDict
import csv
from _csv import reader
from decimal import Decimal, ROUND_UP


##################################### K-NEAREST NEIGHBORS #############################

# calculate knn algorithem, and compute the accuracy on the test file 
def knn(train_set, test_set, k=5):
    correct = 0
    for test_example in test_set:
        train_distances = [two_vec_hamming(train_examples[:-1], test_example[:-1]) for train_examples in train_set]
        max_dist = max(train_distances)
        neighbors = []
        for i in range(k):
            curr_min = train_distances.index(min(train_distances))
            neighbors.append(train_set[curr_min][-1])
            train_distances[curr_min] = max_dist
        pred = Counter(neighbors).most_common(1)[0][0]
        if pred == test_example[-1]:
            correct += 1
    acc = '{:>.02f}'.format(correct / len(test_set))
    return acc

##################################### NAIVE BAYES #####################################

# in order to calculate naive bayes algorithmcreate,
# this function create a data structure that save the probabilities relevant to that algo
# that function return this structure
def naive_bayes(data, attributes, att_classes):
    
    n_examples = len(data[0])
    prediction_counts = Counter(data[-1])
    priors = dict()
    probs_dictionaries = dict()
    indexes_prediction = dict()

    for e in prediction_counts:
        priors[e] = prediction_counts[e]/n_examples
        indexes_prediction[e] = [i for i in range(n_examples) if data[-1][i] == e]

    for p in att_classes[-1]:
        probs_dictionaries[p] = dict()
        for i, att in enumerate(attributes[:-1]):
            for clss in att_classes[i]:
                name = att + '' + clss
                probs_dictionaries[p][name] = (len([j for j in indexes_prediction[p] if data[i][j] == clss]) + 1)\
                                              / (len(indexes_prediction[p]) + len(att_classes[i]))
    return priors, probs_dictionaries

# get predictions and accuracy rate for the test by the naive bayes algorithm
# return prediction on the test set and accuracy
def naive_bayes_test(test_examples, attributes, att_classes, priors, probs_dictionary):

    correct = 0
    classifications = att_classes[-1]
    for x in test_examples:
        pred = None
        max_prob = 0
        for c in classifications:
            prob = priors[c]
            for att_type, att_value in zip(attributes[:-1], x[:-1]):
                prob *= probs_dictionary[c][att_type + '' + att_value]
            if prob > max_prob:
                max_prob = prob
                pred = c
            elif prob == max_prob and (c == 'yes' or c == '1' or c == 'True' or c == 'true'):
                pred = c
        if pred == x[-1]:
            correct += 1
    acc = '{:>.02f}'.format(correct / len(test_examples))
    return acc

##################################### ID3 #############################################

# class Node that contains initiallize function and 
# function that check if is leaf, the values of node 
# and get func to the node value
class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

    def is_leaf_node(self):
        return len(self.children) == 0

    def __repr__(self):
        return str(self.value)

    def __lt__(self, other):
        return self.value < other.value

# read data from the given file in specific way- in order to calculate the id3 algorithem
# returns a list, where each element is a list of features which corresponds to a row from the file
def init_data_from_file(file_name):
        data = []
        with open(file_name, 'r') as tsvfile:
            attribute_field_names = next(tsvfile).strip().split('\t')
            rows = csv.reader(tsvfile, delimiter='\t')
            for row in rows:
                data.append([x.strip() for x in row])
        return data, attribute_field_names[:-1]

# returns a Counter object that contains the total count of the classifications in the given input
# all the classifications is mapped to a count
def get_classification_count(examples):
    classifications_tuple = tuple(get_classifications(examples))
    counter = Counter(classifications_tuple)
    return counter

# get function that return only the lables of each row in the data
def get_classifications(data):
    return [x[-1] for x in data]

# updates the dict with the classification.
# if it is exists, increase by 1
def _update_counting_dict(counting_dict, classification):
    if classification in counting_dict.keys():
        counting_dict[classification] += 1

# get the lables and init a counting dictionary
def _initialize_counting_dict(classifications):
    counting_dict = {}
    for classification in classifications:
        counting_dict[classification] = 0
    return counting_dict

 # create a dictionary of counting dictionary
 # it's maps between all the possible values of the attribute index that given
# and it's classifications in the examples given.
# for example:
# { AttributeValue_1: {Classification_1:Count,
#                      Classification_2:Count},
#   AttributeValue_2: {Classification_1:Count,
#                      Classification_2:Count} }
def get_attribute_values_counting_dicts(attribute_index, examples):
    classifications = set(get_classifications(examples))

    attribute_values_counting_dicts = {}
    for example in examples:
        attribute_value = example[attribute_index]

        if attribute_value not in attribute_values_counting_dicts.keys():
            attribute_values_counting_dicts[attribute_value] = _initialize_counting_dict(classifications)

        _update_counting_dict(attribute_values_counting_dicts[attribute_value], example[-1])
    return attribute_values_counting_dicts

# get list of elements and return the mult result
def multiply_elements(element_list):
    result = 1
    for element in element_list:
        result *= element
    return result


# calculates the gain (decision, attribute) function
#return the gain value
def _get_information_gain(decision_entropy, attribute_index, examples):
    gain = decision_entropy
    attribute_values_counting_dicts = get_attribute_values_counting_dicts(attribute_index, examples)

    for attribute_value, counting_dict in attribute_values_counting_dicts.items():
        entropy, attribute_row_count = _get_entropy_from_counting_dict(counting_dict)

        p = attribute_row_count / len(examples)  # relative factor of attribute in all of the examples
        gain -= p * entropy
    return gain


# returns the entropy calculated from the counting dictionary values
# and the total count of elements in the dictionary 
# return the total sum of all the counters and the entropy value
def _get_entropy_from_counting_dict(counting_dict):
    total_count = sum(counting_dict.values())
    entropy = 0
    for count in counting_dict.values():
        if count != 0:
            relative_count = count / total_count
            entropy += -relative_count * math.log(relative_count, 2)  # log base 2
    return entropy, total_count

# chooses the best attribute to split the decision tree 
# using to the return values of the function above
# returns the index of the 'best' attribute
def _choose_attribute_by_entropy(attribute_indices, examples):
    attributes_gain = []
    classification_count = get_classification_count(examples)
    decision_entropy, total_count = _get_entropy_from_counting_dict(classification_count)

    for attribute_index in attribute_indices:
        gain = _get_information_gain(decision_entropy, attribute_index, examples)
        attributes_gain.append((attribute_index, gain))

    max_element = max(attributes_gain, key=lambda x: x[1])
    return max_element[0]  

# returns the most common classification
def _get_most_common_classification(classifications):
    classifications_tuple = tuple(classifications)
    counter = Counter(classifications_tuple)
    occurrences = sorted(counter.items(), reverse=True)
    return max(occurrences, key=lambda x: x[1])[0]

# returns 1 if all the classifications are the same
def _same_classification_check(classifications):
    return len(set(classifications)) == 1

# creates a decision tree using id3 algorithm and return the tree
def _dtl(examples, attribute_indices, attribute_field_names, classifications, default, original_data):
    if len(examples) == 0:
        return Node(default)
    elif _same_classification_check(classifications):
        # same classification in all the classifications, returns one of them.
        return Node(classifications[0])
    elif len(attribute_indices) == 0:
        return Node(_get_most_common_classification(classifications))
    else:
        best_attribute_index = _choose_attribute_by_entropy(attribute_indices, examples)
        tree = Node(attribute_field_names[best_attribute_index])
        all_possible_attribute_values = set([x[best_attribute_index] for x in original_data])

        for value in sorted(list(all_possible_attribute_values)):
            next_examples = [example for example in examples if value == example[best_attribute_index]]
            next_attributes_indices = attribute_indices.copy()
            next_attributes_indices.remove(best_attribute_index)
            next_classifications = get_classifications(next_examples)
            next_default = _get_most_common_classification(classifications)

            branch = Node(value)
            sub_tree = _dtl(next_examples, next_attributes_indices, attribute_field_names, next_classifications,
                            next_default, original_data)
            branch.children.append(sub_tree)
            tree.children.append(branch)
        return tree

# create the tree by the given data and attributes names
def create_decision_tree(data, attribute_field_names):
    classifications = get_classifications(data)
    default = _get_most_common_classification(classifications)
    attribute_indices = [i for i in range(len(attribute_field_names))]
    decision_tree = _dtl(data, attribute_indices, attribute_field_names, classifications, default, data)

    output_tree_to_file(OUTPUT_FILE, decision_tree, attribute_field_names)

    return decision_tree

# predict the classification of each row in the data 
# return a list of the presiction
def predict(decision_tree_root, data, attribute_field_names):
    prediction_list = []
    for data_row in data:
        classification = _get_row_prediction(decision_tree_root, data_row, attribute_field_names)
        data_row[-1] = classification
        prediction_list.append(data_row)
    return prediction_list

# function tht return the prediction of a row
def _get_row_prediction(decision_tree_root, data_row, attribute_field_names):
    return _traverse_tree(decision_tree_root, data_row, 0, attribute_field_names)

# a recursive function to search in the tree
#  and assign classification if it's a leaf
def _traverse_tree(decision_tree_root, data_row, attribute_index, attribute_field_names):
    if decision_tree_root.is_leaf_node():
        return decision_tree_root.value
    else:
        if decision_tree_root.value in attribute_field_names:
            attribute_index = attribute_field_names.index(decision_tree_root.value)
        data_row_value = data_row[attribute_index]
        next_node = _get_next_node(decision_tree_root, data_row_value, attribute_field_names)
        return _traverse_tree(next_node, data_row, attribute_index, attribute_field_names)

# get the next node in the tree in the cases in the if statment
#if it's a leaf return none
def _get_next_node(current_node, value, attribute_field_names):
    for child in current_node.children:
        if child.value == value or child.value in attribute_field_names or child.is_leaf_node():
            return child
    return None

# write the decision tree to the file name given
def output_tree_to_file(file_name, decision_tree, attribute_field_names):
    output_strings = []
    _write_to_file(output_strings, 0, decision_tree, attribute_field_names)
    output_strings[-1] = output_strings[-1].rstrip('\n')
    with open(file_name, 'w') as output_file:
        output_file.writelines(output_strings)

# recursive function to write the tree file
def _write_to_file(output_strings, depth, tree, attribute_field_names):
    tree.children = sorted(tree.children)
    for child in tree.children:
        if child.is_leaf_node():
            output_strings.append(":" + child.value + "\n")
            break
        if tree.value in attribute_field_names:
            line_break_str = ""
            if depth != 0:
                line_break_str = "\t" * int(depth / 2) + "|"
            output_strings.append(line_break_str + tree.value + "=" + child.value)
            if not child.children[0].is_leaf_node():
                output_strings.append("\n")
        _write_to_file(output_strings, depth + 1, child, attribute_field_names)

# calculate and return the accuracy after running the id3 algo 
def _calculate_accuracy(classifications_list,file_name):
    count_correct = 0
    correct_classifications = []
    with open(file_name, 'r') as tsvfile:
        next(tsvfile).strip().split('\t')
        rows = reader(tsvfile, delimiter='\t')
        for row in rows:
            correct_classifications.append(row[-1])  # only classification

    for classification, correct in zip(classifications_list, correct_classifications):
        if classification == correct:
            count_correct += 1

    accuracy = count_correct / float(len(correct_classifications))
    # return Decimal(accuracy).quantize(Decimal('.00'), rounding=ROUND_UP)
    return round(accuracy + 0.005, 2)

##################################### UTILITYS FUNCTIONS #####################################

# read the data from the given file  
def read_data_from_file (filename):
  
    file = open(filename)
    all_file_lines = file.readlines()
    attributes = [w.strip() for w in all_file_lines[0].split('\t')]
    n_att = len(attributes)
    data_examples = [[w.strip() for w in line.split('\t')] for line in all_file_lines[1:]]
    data_attributes = [[example[i] for example in data_examples] for i in range(n_att)]
    att_classes_names = [set(classes) for classes in data_attributes]
    attribute_without_classifications = attributes[:-1]

    return data_examples, data_attributes, attributes, att_classes_names,attribute_without_classifications

# calculate hamming distance according to the formula
def two_vec_hamming(vec1, vec2):
    return len([i for i, j in zip(vec1, vec2) if i != j])

# divide the data to 5 parts in our case 
# and return all that parts
def divide_data(data):
    parts = {}
    length = len(data)
    for i in range(5):
        if (i == (5 - 1)):
            start = (length // 5) * i
            parts[i] = data[start:]
        else:
            start = (length // 5) * i
            end = (length // 5) * (i + 1)
            parts[i] = data[start:end]

    return parts   

# from the parts that return from the func above
# devide the data to train and test chuncks
def create_train_and_test(index, parts):
    train_data = []
    test_data = []
    # Create the train data
    for i in range(5):
        if (i != index):
            for example in parts[i]:
                train_data.append(example)
        else:
            for example in parts[i]:
                test_data.append(example)

    return train_data, test_data

# write the data to a given file
def write_data_to_file(data,attributes,file_name):
        with open(file_name, 'w') as output_file:
            att_line = ""
            for att in attributes:
                att_line = att_line +str (att) +'\t'            
            output_file.write(att_line.strip()) 
            output_file.write('\n')
            entry_line = ""

            for entry in data:
                entry_line=""           
                for i in entry:
                    entry_line = entry_line + i +'\t' 
                output_file.write(entry_line.strip())
                output_file.write('\n')

# create 4 file of train and 4 files of test from the given train file    
def create_k_train_and_test_files():
    data_examples, data_attributes, attributes, att_classes_names, attr_without_class = read_data_from_file(TRAIN)
    parts = divide_data(data_examples)
    train_files = ['train_1.txt', 'train_2.txt', 'train_3.txt','train_4.txt','train_5.txt']
    test_files = ['test_1.txt', 'test_2.txt', 'test_3.txt','test_4.txt','test_5.txt']
    
    for i in range(5):
        train_data, test_data = create_train_and_test(i,parts)
        write_data_to_file(train_data,attributes,train_files[i])
        write_data_to_file(test_data,attributes,test_files[i])

    return train_files,test_files        

# calculate the accuracy of each algorithem 
# by using k fold cross validation, k=5 in our case
def calc_algos_accuracy_by_kfold(k = 5):
    data_examples, data_attributes, attributes, att_classes_names, attr_without_class = read_data_from_file(TEST)
    parts = divide_data(data_examples)
    all_knn_acc = []
    all_nb_acc = []
    for i in range (k):
        train_data, test_data = create_train_and_test(i,parts)
        knn_acc = knn(train_data,test_data)
        all_knn_acc.append(float(knn_acc))
        
        priors, probs_dictionaries = naive_bayes(data_attributes, attributes, att_classes_names)
        nb_acc = naive_bayes_test(test_data, attributes, att_classes_names, priors, probs_dictionaries)
        all_nb_acc.append(float(nb_acc))
        
    knn_acc_sum = math.fsum(all_knn_acc)
    knn_avg_acc = knn_acc_sum / k
    nb_acc_sum =  math.fsum(all_nb_acc)
    nb_avg_acc = nb_acc_sum / k

    train_files, test_files= create_k_train_and_test_files()
    
    all_dt_acc = []
    for i in range (k) : 
        classifications_ordered_dict = OrderedDict()
        train_data, attribute_field_names = init_data_from_file(train_files[i])
        tree = create_decision_tree(train_data, attribute_field_names)
        to_predict, attribute_field_names = init_data_from_file(test_files[i])
        prediction_list = predict(tree, to_predict, attribute_field_names)
        classifications_ordered_dict['DT'] = get_classifications(prediction_list)
        dt_acc =_calculate_accuracy(list(classifications_ordered_dict.values())[0],test_files[i])
        all_dt_acc.append(dt_acc)
    dt_acc_sum = math.fsum(all_dt_acc)  
    dt_avg_acc = dt_acc_sum / k

    return knn_avg_acc, nb_avg_acc, dt_avg_acc

# write the given valuse of accuracy to the output file
def write_output_file(acc_dt, acc_knn, acc_nb):
    output = open(OUTPUT_FILE, 'a')
    output.write ("\n\n")
    output.write("{:.2f}".format(dt_acc) + '\t' + "{:.2f}".format(float(knn_acc)) + '\t' + "{:.2f}".format(float(nb_acc)))    
    output.close()

# write only the given valuse of accuracy to the output file            
def write_accuracy_to_file(dt_acc,knn_acc,nb_acc):
    with open(ACCURACY_FILE, 'w') as file:
        file.write("{:.2f}".format(dt_acc) + '\t' + "{:.2f}".format(float(knn_acc)) + '\t' + "{:.2f}".format(float(nb_acc)))    
                    
##################################### MAIN FUNCTION #####################################

if __name__ == '__main__':

    TRAIN = 'train.txt'
    TEST = 'test.txt'
    ACCURACY_FILE = 'accuracy.txt'
    OUTPUT_FILE ='output.txt'
    
    # create orderd dictionary
    classifications_ordered_dict = OrderedDict()

    # init data from to the id3 algo train file 
    train_data, attribute_field_names = init_data_from_file(TRAIN)
    tree = create_decision_tree(train_data, attribute_field_names)

    # read data from train and test files
    train_examples, train_data_attributes, train_attributes, att_train_names, attr_train_without_class = read_data_from_file(TRAIN)
    test_examples, test_data_attributes, test_attributes, att_test_names, attr_test_without_class = read_data_from_file(TEST)

    # dt
    to_predict, attribute_field_names = init_data_from_file(TEST)
    prediction_list = predict(tree, to_predict, attribute_field_names)
    classifications_ordered_dict['DT'] = get_classifications(prediction_list)
    dt_acc=_calculate_accuracy(list(classifications_ordered_dict.values())[0],TEST)

    # knn
    knn_acc = knn(train_examples,test_examples)

    # nb
    priors, probs_dictionaries = naive_bayes(train_data_attributes, train_attributes, att_train_names)
    nb_acc = naive_bayes_test(test_examples, test_attributes, att_test_names, priors, probs_dictionaries)

    # output tree & accuracy to output file 
    write_output_file(dt_acc, knn_acc, nb_acc)
