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
#include <string>

using namespace std;

class Node {
    int x;
    int y;
    char z;
    int k;
public:
    Node();

    Node(int, int, char, char);

    int get_key() const;

    void print() const;
};

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

class BTree {
    BNode *root;
    int min_degree;
public:
    BTree(int);

    void print();

    BNode *search(int);

    void insert(int, Node *);
};

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

Node::Node() {
    this->x = -1;
    this->y = -1;
    this->z = -1;
    this->k = -1;
}

Node::Node(int x, int y, char z, char key) {
    this->x = x;
    this->y = y;
    this->z = z;
    switch (key) {
        case 'x':
            this->k = x;
            break;
        case 'y':
            this->k = y;
            break;
        case 'z':
            this->k = (int) (u_char) z;
            break;
        default:
            cout << "Given key " << key << " is not valid!" << endl;
    }
}

void Node::print() const {
    cout << "("
         << x << ","
         << y << ","
         << z << ")";
}

int Node::get_key() const {
    return k;
}

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
            BNode *bnode = new BNode(min_degree, false);
            bnode->child[0] = root;
            bnode->split_child(0, root);

            if (bnode->keys[0] < key) {
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