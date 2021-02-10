//
// Created by batuhanfaik on 08/02/2021.
//
#include <iostream>
#include "node.h"

using namespace std;


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
            this->k = (int)(u_char) z;
            break;
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
