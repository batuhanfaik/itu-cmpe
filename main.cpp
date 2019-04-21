//
// Created by;
// Batuhan Faik Derinbay 
// 150180705
// on 4/21/19.
//
#include "assignment2.cpp"

int main(){
    OperatorGrid a(10,7);
    ArithmeticOperator *x = new ArithmeticOperator(2,2,1,'x');
    ArithmeticOperator *y = new ArithmeticOperator(2,4,1,'+');
    ArithmeticOperator *z = new ArithmeticOperator(2,6,1,'x');
    ArithmeticOperator *t = new ArithmeticOperator(2,1,9,'/');
    ArithmeticOperator *w = new ArithmeticOperator(9,4,3,'-');
    ArithmeticOperator *xx = new ArithmeticOperator(5,4,3,'/');
    ArithmeticOperator *yy = new ArithmeticOperator(5,4,3,'-');
    ArithmeticOperator *zz = new ArithmeticOperator(5,2,1,'-');
    ArithmeticOperator *tt = new ArithmeticOperator(7,6,1,'/');
    ArithmeticOperator *ww = new ArithmeticOperator(5,6,1,'+');
    a.place_operator(x);
    a.place_operator(y);
    a.place_operator(z);
    a.place_operator(t);
    a.place_operator(w);
    a.place_operator(xx);
    a.place_operator(yy);
    a.place_operator(zz);
    a.place_operator(tt);
    a.place_operator(ww);
    a.move_operator(1,1,'D',2);
    a.move_operator(5,1,'D',5);
    a.move_operator(1,1,'D',2);
    a.move_operator(2,4,'L',1);
    a.move_operator(2,4,'L',2);
    a.move_operator(8,5,'L',3);
    a.move_operator(1,7,'D',5);
    a.move_operator(10,1,'R',7);
    a.move_operator(5,4,'U',3);
    a.print_operators();
    return 0;
}