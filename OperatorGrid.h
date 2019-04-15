//
// Created by;
// Batuhan Faik Derinbay 
// 150180705
// on 4/15/19.
//
#ifndef OOP_HW2_OPERATORGRID_H
#define OOP_HW2_OPERATORGRID_H

#include "ArithmeticOperator.h"

class OperatorGrid{
    int grid_rows;
    int grid_cols;
    char **grid;

    int num_operators;
    ArithmeticOperator *operators[MAX_OPERATION_SIZE];
public:
    OperatorGrid(int rows, int cols); //Default constructor
    ~OperatorGrid();

    bool place_operator(ArithmeticOperator *); //Places an operator
    bool move_operator(int x, int y, char direction, int move_by); //Moves the operator
    void print_operators(); //Prints the operator
};

#endif //OOP_HW2_OPERATORGRID_H
