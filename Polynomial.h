//
// Created by;
// Batuhan Faik Derinbay 
// 150180705
// on 3/17/19.
//
#ifndef OOP_HW1_POLYNOMIAL_H
#define OOP_HW1_POLYNOMIAL_H

#include <iostream>
using namespace std;

class Polynomial {
    int degree;
    int* coef_array;
//    int polynomial_no;
//    static int polynomials_created;
public:
    // Default constructor
    Polynomial();
    // Constructor
    Polynomial(int,const int*);
    // Copy constructor
    Polynomial(const Polynomial&);
    // Getter methods
    int getDegree() const;
    int getCoefArray(int) const;
//    int getPolyNo() const;
    // Operator overload +
    Polynomial operator+(const Polynomial&) const;
    // Operator overload *
    Polynomial operator*(const Polynomial&) const;
    // Operator overload <<
    friend ostream& operator<<(ostream&, const Polynomial&);
    // Temporary print
    const void print() const;
//    // Clear polynomial counter
//    static void clear_counter(){polynomials_created = 0;};
    ~Polynomial();
};


#endif //OOP_HW1_POLYNOMIAL_H
