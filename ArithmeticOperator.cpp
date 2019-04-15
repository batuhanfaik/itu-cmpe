//
// Created by;
// Batuhan Faik Derinbay 
// 150180705
// on 4/15/19.
//
#include "ArithmeticOperator.h"
#include <iostream>
#include <string>

using namespace std;

ArithmeticOperator::ArithmeticOperator(int x, int y, int size, char sign):Operator(x,y,size) {
    if (sign == '+' || sign == '-' || sign == '*' || sign == '/'){
        this->sign = sign;
    }
    else {
        cout << "SIGN parameter is invalid!" << endl;
    }
}

char ArithmeticOperator::get_sign() { //Getter of the sign
    return sign;
}

void ArithmeticOperator::print_operator() {
    cout << "ARITHMETIC_OPERATOR[" << sign << "], CENTER_LOCATION[" << get_x() << "," << get_y()
    << "], SIZE[" << get_size() << "]" << endl;
}
