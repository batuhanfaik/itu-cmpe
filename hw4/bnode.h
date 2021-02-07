//
// Created by batuhanfaik on 07/02/2021.
//

#ifndef HW4_BNODE_H
#define HW4_BNODE_H

#include "node.h"

using namespace std;

class BNode {
    Node *nodes;
    BNode **child;
    int min_degree;
    int n_key;
    bool leaf;
public:
    BNode();

    friend class BTree;
};


#endif //HW4_BNODE_H
