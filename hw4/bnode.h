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

    int find_key(int);

    void remove(int);

    void remove_from_leaf(int);

    void remove_from_nonleaf(int);

    int get_predecessor(int);

    int get_successor(int);

    void fill(int);

    void borrow_from_prev(int);

    void borrow_from_next(int);

    void merge(int);

    friend class BTree;
};


#endif //HW4_BNODE_H
