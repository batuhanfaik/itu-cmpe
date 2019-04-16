//
// Created by;
// Batuhan Faik Derinbay 
// 150180705
// on 4/16/19.
//

#include <iostream>
#include "Operator.h"
#include "ArithmeticOperator.h"
#include "OperatorGrid.h"

using namespace std;

int main(){
    OperatorGrid my_grid(7,10);
    ArithmeticOperator ao1(3,3,2,'+');
    ArithmeticOperator ao2(5,5,1,'/');
    ArithmeticOperator ao3(5,6,1,'/');
    ArithmeticOperator ao4(5,7,0,'x');
    my_grid.place_operator(&ao1);
    my_grid.place_operator(&ao2);
    my_grid.place_operator(&ao3);
    my_grid.place_operator(&ao4);
    cout << "Before move" << endl;
    my_grid.print_operators();
    my_grid.move_operator(6,4,'R',2);
    cout << "After move" << endl;
    my_grid.print_operators();
}