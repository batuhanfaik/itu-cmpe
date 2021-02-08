//
// Created by batuhanfaik on 07/02/2021.
//

#ifndef HW4_BNODE_H
#define HW4_BNODE_H

#include "node.h"

using namespace std;

class BNode {
    int *keys;
    Node **nodes;
    BNode **child;
    int min_degree;
    int n_key;
    bool leaf;
public:
    BNode(int, bool);

    void insert_nonfull(int, Node *);

    void split_child(int, BNode *);

    void traverse();

    BNode *search(int);

    friend class BTree;
};


#endif //HW4_BNODE_H
