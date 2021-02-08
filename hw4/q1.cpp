/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * /
* @ Filename: q1
* @ Date: 07-Feb-2021
* @ AUTHOR: batuhanfaik
* @ Copyright (C) 2020 Batuhan Faik Derinbay
* @ Project: hw4
* @ Description: B Tree Creation and Insertion
* @ Compiling: g++ -o a.out q1.cpp
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include <iostream>
#include <fstream>
#include <string>

#include "btree.h"

using namespace std;

int main() {
    ifstream file;
    file.open("sample2.txt");

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
//        cout << key_x << " " << key_y << " " << key_z << endl;

        Node *node_tmp = new Node(key_x, key_y, key_z, sorting_key);
        btree.insert(node_tmp->get_key(), node_tmp);
    }

//    cout << endl << node_count << " " << tree_degree << " " << sorting_key << endl;
    btree.traverse();

    file.close();
    return 0;
}