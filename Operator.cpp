//
// Created by;
// Batuhan Faik Derinbay
// 150180705
// on 4/15/19.
//
#include "Operator.h"

Operator::Operator(int x, int y, int size) {
    center_x = x;
    center_y = y;
    op_size = size;
}

void Operator::reset(int new_x, int new_y, int new_size) {
    //Reassign all values
    center_x = new_x;
    center_y = new_y;
    op_size = new_size;
}

void Operator::set_x(int new_x) { //Setter of x
    center_x = new_x;
}
int Operator::get_x() { //Getter of x
    return center_x;
}

void Operator::set_y(int new_y) { //Setter of y
    center_y = new_y;
}
int Operator::get_y() { //Getter of y
    return center_y;
}

void Operator::set_size(int new_size) { //Setter of size
    op_size = new_size;
}
int Operator::get_size() { //Getter of size
    return op_size;
}