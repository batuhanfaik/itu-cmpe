//
// Created by;
// Batuhan Faik Derinbay 
// 150180705
// on 4/15/19.
//
#ifndef OOP_HW2_OPERATOR_H
#define OOP_HW2_OPERATOR_H

#include "ArithmeticOperator.h"

class Operator{
    int center_x;
    int center_y;
    int op_size;
public:
    Operator(int x, int y, int size); //Default constructor

    void reset(int new_x, int new_y, int new_size); //Reset position

    void set_x(int new_x); //Setter of position x
    int get_x(); //Getter of position x

    void set_y(int new_y); //Setter of position y
    int get_y(); //Getter of position y

    void set_size(int new_size); //Setter of the operator size
    int get_size(); //Getter of the operator size
};

#endif //OOP_HW2_OPERATOR_H
