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
    int node_count, tree_degree, key_x, key_y;
    char sorting_key, key_z;
    string line;
    cin >> node_count;
    cin >> tree_degree;
    cin >> sorting_key;

    BTree btree(tree_degree);

    for (int i = 0; i < node_count; ++i) {
        cin >> key_x >> key_y >> key_z;

        Node *node_tmp = new Node(key_x, key_y, key_z, sorting_key);
        btree.insert(node_tmp->get_key(), node_tmp);
    }

    btree.print();
    return 0;
}