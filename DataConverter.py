'''
Sam Congdon, Kendall Dilorenzo, Michel Hewitt
CSCI 447: MachineLearning
Project 4: Data Converter
December 11, 2017

This python module is used to convert csv or tsv data files into formats usable for clustering.
Nominal attributes are encoded into ordinal values determined by user input.
The converted data is output in csv format, with each individual's data on a single line
'''

import re

def get_column_values(column_number, data):
    ''' Determine all possible values a column of data holds in order to determine
        a nominal data type. '''

    value_list = {}     # Dictionary to hold all the classes present in the data

    # checks the value contained in column in each line, adding new values to the dict
    for line in data.split('\n'):
        line = line.split(',')
        if len(line) > 1: value_list.update({line[column_number]: 1})

    try:
        # if the column contains only real values the function returns true
        for key in value_list.keys():
            float(key)
        return (True, value_list)
    except ValueError:
        # else returns false and a dictionary of nominal attributes
        return (False, value_list)

def real_number_encode(dict):
    ''' Creates a dictionary to encode nominal attributes with real values, requires user input'''
    print('Attributes in column: {}'.format(dict))
    for key in dict.keys():
        dict[key] = float(input('Enter the ordinal integer value of attribute {}:'.format(key)))
    return dict


def process_file(name_in, name_out, remove_column):
    ''' Takes input file of a csv or tsv format and formats the value into real values usable for clustering.
        Requires user input to determine the integer values of the ordinal variables. Outputs each data point
        on a single line, comma separated. '''

    fin = open(name_in, 'r')
    fout = open(name_out, 'w')
    data = ''

    column_types = []
    encoder = []

    for line in fin:
        line = re.sub("\s+", ',', line.strip())  # strips all white space, adding commas to separate values
        data += line + '\n'  # adds the converted line to the data string

    # used to remove the specified columns from the data
    if remove_column[0] != None:
        old_data = data
        data = ''
        for line in old_data.split('\n'):
            line = line.split(',')
            if len(line) > len(remove_column):
                for column in remove_column:
                    line.pop(column)
                data += ','.join(line) + '\n'

    sample = data.split('\n')
    sample = sample[0].split(',')

    for column in range(len(sample)):
        # determine whether the attributes must be encoded
        real, dict = get_column_values(column, data)
        column_types.append(real)

        # if encoding is needed, do it
        if not real:
            real_number_encode(dict)
            encoder.append(dict.copy())
        # else move on
        else:
            encoder.append({}.copy())

    # now we create the output files with the encoder
    for line in data.split('\n'):
        line = line.split(',')
        output = ''
        # add the input values on a line
        for i in range(len(line)):
            if column_types[i]: output += '{},'.format(line[i])
            elif line[i] != '': output += '{},'.format(encoder[i][line[i]])
        fout.write(output[:-1] + '\n')

    fin.close()
    fout.close()

# process each of the data files, remove the individual's class in classification problems
process_file('datasets/iris.csv', 'datasets/iris.txt', [-1])

process_file('datasets/abalone.csv', 'datasets/abalone.txt', [-1])
process_file('datasets/car.csv', 'datasets/car.txt', [-1])
process_file('datasets/cmc.csv', 'datasets/cmc.txt', [-1])
process_file('datasets/synthetic_control.tsv', 'datasets/synthetic_control.txt', [None])
process_file('datasets/yeast.tsv', 'datasets/yeast.txt', [0, -1])

# data has missing attributes
# process_file('datasets/coil_1999_competition.csv', 'datasets/coil_1999_competition.txt', [], [None])
