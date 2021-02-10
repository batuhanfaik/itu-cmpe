//
// Created by batuhanfaik on 07/02/2021.
//

#ifndef HW4_BTREE_H
#define HW4_BTREE_H

#include "bnode.h"

using namespace std;

class BTree {
    BNode *root;
    int min_degree;
public:
    BTree(int);

    void print();

    BNode *search(int);

    void insert(int, Node*);

    void remove(int);
};


#endif //HW4_BTREE_H
