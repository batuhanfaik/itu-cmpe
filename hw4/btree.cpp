//
// Created by batuhanfaik on 07/02/2021.
//

#include "node.h"
#include "btree.h"

BTree::BTree(int min_degree) {
    this->root = nullptr;
    this->min_degree = min_degree;
}

void BTree::print() {
    if (root != nullptr) {
        root->traverse();
    }
}

BNode *BTree::search(int key) {
    if (root != nullptr) {
        return root->search(key);
    }
    return nullptr;
}

void BTree::insert(int key, Node *node_ptr) {
    if (!root) {
        root = new BNode(min_degree, true);
        root->keys[0] = key;
        root->nodes[0] = node_ptr;
        root->n_key = 1;
    } else {
        int max_degree = 2 * min_degree - 1;
        if (root->n_key == max_degree) {
            BNode* bnode = new BNode(min_degree, false);
            bnode->child[0] = root;
            bnode->split_child(0, root);

            if (bnode->keys[0] < key){
                bnode->child[1]->insert_nonfull(key, node_ptr);
            } else {
                bnode->child[0]->insert_nonfull(key, node_ptr);
            }

            root = bnode;
        } else {
            root->insert_nonfull(key, node_ptr);
        }
    }
}

