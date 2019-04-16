//
// Created by;
// Batuhan Faik Derinbay 
// 150180705
// on 4/15/19.
//
#ifndef OOP_HW2_ARITHMETICOPERATOR_H
#define OOP_HW2_ARITHMETICOPERATOR_H

#include "Operator.h"

class ArithmeticOperator: public Operator {
    char sign;
public:
    ArithmeticOperator(); //Default constructor

    ArithmeticOperator(int x, int y, int size, char sign); //Constructor

    char get_sign(); //Getter method of the sign

    void print_operator(); //Prints out operator's center location, size and sign character
};

#endif //OOP_HW2_ARITHMETICOPERATOR_H
