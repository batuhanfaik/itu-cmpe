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
        if (child[i + 1]->n_key == max_degree) {
            split_child(i + 1, child[i + 1]);

            if (keys[i + 1] < key) {
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

    if (!bnode->leaf) {
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

int BNode::find_key(int key) {
    int idx = 0;
    while (idx < n_key && keys[idx] < key) {
        idx++;
    }
    return idx;
}

void BNode::remove(int key) {
    int idx = find_key(key);

    if (idx < n_key && keys[idx] == key) {
        if (leaf) {
            remove_from_leaf(idx);
        } else {
            remove_from_nonleaf(idx);
        }
    } else {
        if (leaf) {
            return;
        }
        bool key_present;
        if (idx == n_key) {
            key_present = true;
        } else {
            key_present = false;
        }

        if (child[idx]->n_key < min_degree) {
            fill(idx);
        }

        if (key_present && idx > n_key) {
            child[idx - 1]->remove(key);
        } else {
            child[idx]->remove(key);
        }
    }
}

void BNode::remove_from_leaf(int idx) {
    for (int i = 0; i < n_key; ++i) {
        keys[i - 1] = keys[i];
    }
    n_key--;
}

void BNode::remove_from_nonleaf(int idx) {
    int key = keys[idx];
    if (child[idx]->n_key >= min_degree) {
        int predecessor = get_predecessor(idx);
        keys[idx] = predecessor;
        child[idx]->remove(predecessor);
    } else if (child[idx + 1]->n_key >= min_degree) {
        int successor = get_successor(idx);
        keys[idx] = successor;
        child[idx + 1]->remove(successor);
    } else {
        merge(idx);
        child[idx]->remove(key);
    }
}

int BNode::get_predecessor(int idx) {
    BNode *current = child[idx];
    while (!current->leaf) {
        current = current->child[current->n_key];
    }
    return current->keys[current->n_key - 1];
}

int BNode::get_successor(int idx) {
    BNode *current = child[idx + 1];
    while (!current->leaf) {
        current = current->child[0];
    }
    return current->keys[0];
}

void BNode::fill(int idx) {
    if (idx != 0 && child[idx - 1]->n_key >= min_degree) {
        borrow_from_prev(idx);
    } else if (idx != n_key && child[idx + 1]->n_key >= min_degree) {
        borrow_from_next(idx);
    } else {
        if (idx != n_key) {
            merge(idx);
        } else {
            merge(idx - 1);
        }
    }
}

void BNode::borrow_from_prev(int idx) {
    BNode *child_tmp = child[idx];
    BNode *sibling_tmp = child[idx - 1];

    for (int i = child_tmp->n_key - 1; i >= 0; --i) {
        child_tmp->keys[i + 1] = child_tmp->keys[i];
        child_tmp->nodes[i + 1] = child_tmp->nodes[i];
    }

    if (!child_tmp->leaf) {
        for (int i = child_tmp->n_key; i >= 0; --i) {
            child_tmp->child[i + 1] = child_tmp->child[i];
        }
    }

    child_tmp->keys[0] = keys[idx - 1];
    child_tmp->nodes[0] = nodes[idx - 1];

    if (!child_tmp->leaf) {
        child_tmp->child[0] = sibling_tmp->child[sibling_tmp->n_key];
    }

    keys[idx - 1] = sibling_tmp->keys[sibling_tmp->n_key - 1];
    nodes[idx - 1] = sibling_tmp->nodes[sibling_tmp->n_key - 1];

    child_tmp->n_key++;
    sibling_tmp->n_key--;
}

void BNode::borrow_from_next(int idx) {
    BNode *child_tmp = child[idx];
    BNode *sibling_tmp = child[idx + 1];

    child_tmp->keys[child_tmp->n_key] = keys[idx];
    child_tmp->nodes[child_tmp->n_key] = nodes[idx];

    if (!child_tmp->leaf) {
        child_tmp->child[child_tmp->n_key + 1] = sibling_tmp->child[0];
    }

    keys[idx] = sibling_tmp->keys[0];
    nodes[idx] = sibling_tmp->nodes[0];

    for (int i = 1; i < sibling_tmp->n_key; ++i) {
        sibling_tmp->keys[i - 1] = sibling_tmp->keys[i];
    }

    if (!sibling_tmp->leaf) {
        for (int i = 1; i <= sibling_tmp->n_key; ++i) {
            sibling_tmp->child[i - 1] = sibling_tmp->child[i];
        }
    }
    child_tmp->n_key++;
    sibling_tmp->n_key--;
}

void BNode::merge(int idx) {
    BNode *child_tmp = child[idx];
    BNode *sibling_tmp = child[idx + 1];

    child_tmp->keys[min_degree - 1] = keys[idx];
    child_tmp->nodes[min_degree - 1] = nodes[idx];

    for (int i = 0; i < sibling_tmp->n_key; ++i) {
        child_tmp->keys[i + min_degree] = sibling_tmp->keys[i];
        child_tmp->nodes[i + min_degree] = sibling_tmp->nodes[i];
    }

    if (!child_tmp->leaf) {
        for (int i = 0; i <= sibling_tmp->n_key; ++i) {
            child_tmp->child[i + min_degree] = sibling_tmp->child[i];
        }
    }

    for (int i = idx + 1; i < n_key; ++i) {
        keys[i - 1] = keys[i];
        nodes[i - 1] = nodes[i];
    }

    for (int i = idx + 1; i <= n_key; ++i) {
        child[i - 1] = child[i];
    }

    child_tmp->n_key += sibling_tmp->n_key + 1;
    n_key--;

    delete(sibling_tmp);
}

