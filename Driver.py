
import os
import random
import DBScan

def import_data(file_name):
    ''' Imports data points from supplied file, formatting them into data point lists to be clustered. '''
    fin = open(file_name, 'r')
    input_line = fin.readline()
    data = []

    while input_line:
        input_line = input_line.strip().split(',')
        if input_line == ['']: break
        for i in range(len(input_line)): input_line[i] = float(input_line[i])
        data.append((input_line))
        input_line = fin.readline()
    fin.close()

    return data, os.path.splitext(file_name)[0][9:] # last return is file name, assumes file is in datasets/ directory


def main():
    data, name = import_data('datasets/iris.txt')

    clusters = DBScan.db_clustering(data, 30)



if __name__ == '__main__':
    main()