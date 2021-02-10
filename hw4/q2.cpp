/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * /
* @ Filename: q2
* @ Date: 07-Feb-2021
* @ AUTHOR: batuhanfaik
* @ Copyright (C) 2020 Batuhan Faik Derinbay
* @ Project: hw4
* @ Description: B Tree Node Removal
* @ Compiling: g++ -o a.out q2.cpp
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include <iostream>
#include <fstream>
#include <string>

#include "btree.h"

using namespace std;

int main() {
    ifstream file;
    file.open("sample1_deletion.txt");

    if (!file) {
        cerr << "File cannot be opened!";
        exit(1);
    }

    int node_count, tree_degree, key_x, key_y;
    char sorting_key, key_z;
    string line;
    file >> node_count;
    file >> tree_degree;
    file >> sorting_key;
    getline(file, line, '\n');

    BTree btree(tree_degree);

    for (int i = 0; i < node_count; ++i) {
        file >> key_x >> key_y >> key_z;
        getline(file, line, '\n');

        Node *node_tmp = new Node(key_x, key_y, key_z, sorting_key);
        btree.insert(node_tmp->get_key(), node_tmp);
    }

    int deletion_key;
    if (sorting_key == 'z'){
        char tmp_key;
        file >> tmp_key;
        deletion_key = (int)(u_char) tmp_key;
    } else {
        file >> deletion_key;
    }

    btree.remove(deletion_key);
    btree.print();

    file.close();
    return 0;
}