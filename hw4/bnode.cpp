//
// Created by batuhanfaik on 07/02/2021.
//

#include <iostream>
#include "node.h"
#include "bnode.h"

using namespace std;

BNode::BNode(int min_degree, bool leaf) {
    this->min_degree = min_degree;
    this->leaf = leaf;
    this->keys = new int[2 * min_degree - 1];
    this->nodes = new Node *[2 * min_degree - 1];
    this->child = new BNode *[2 * min_degree];
    n_key = 0;
}

void BNode::insert_nonfull(int key, Node *node) {
    int i = n_key - 1;
    int max_degree = 2 * min_degree - 1;

    if (leaf) {
        while (i >= 0 && keys[i] > key) {
            keys[i + 1] = keys[i];
            nodes[i + 1] = nodes[i];
            i--;
        }
        keys[i + 1] = key;
        nodes[i + 1] = node;
        n_key++;
    } else {
        while (i >= 0 && keys[i] > key) {
            i--;
        }
        if (child[i + 1]->n_key == max_degree){
            split_child(i + 1, child[i + 1]);
            
            if (keys[i + 1] < key){
                i++;
            }
        }
        child[i + 1]->insert_nonfull(key, node);
    }
}

void BNode::split_child(int idx, BNode *bnode) {
    BNode *bnode_new = new BNode(bnode->min_degree, bnode->leaf);
    bnode_new->n_key = min_degree - 1;

    for (int i = 0; i < min_degree - 1; i++) {
        bnode_new->keys[i] = bnode->keys[i + min_degree];
        bnode_new->nodes[i] = bnode->nodes[i + min_degree];
    }

    if (!bnode->leaf){
        for (int i = 0; i < min_degree; i++) {
            bnode_new->child[i] = bnode->child[i + min_degree];
        }
    }
    bnode->n_key = min_degree - 1;

    for (int i = n_key; i >= idx + 1; i--) {
        child[i + 1] = child[i];
    }
    child[idx + 1] = bnode_new;
    for (int i = n_key - 1; i >= idx; i--) {
        keys[i + 1] = keys[i];
        nodes[i + 1] = nodes[i];
    }
    keys[idx] = bnode->keys[min_degree - 1];
    nodes[idx] = bnode->nodes[min_degree - 1];

    n_key++;
}

//void BNode::traverse() {
//    int i;
//    for (i = 0; i < n_key; i++) {
//        nodes[i]->print();
//    }
//    cout << endl;
//
//    if (!leaf) {
//        child[i]->traverse();
//    }
//}
void BNode::traverse() {
    int i;
    for (int j = 0; j < n_key; ++j) {
        nodes[j]->print();
    }
    cout << endl;
    for (i = 0; i < n_key; i++) {
        if (!leaf) {
            child[i]->traverse();
        }
    }
    if (!leaf) {
        child[i]->traverse();
    }
}

BNode *BNode::search(int key) {
    int i = 0;
    while (i < n_key && key > keys[i]) {
        i++;
    }

    if (keys[i] == key) {
        return this;
    }
    if (leaf) {
        return nullptr;
    }
    return child[i]->search(key);
}
