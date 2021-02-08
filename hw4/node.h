//
// Created by batuhanfaik on 07/02/2021.
//

#ifndef HW4_NODE_H
#define HW4_NODE_H


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


#endif //HW4_NODE_H
