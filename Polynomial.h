//
// Created by;
// Batuhan Faik Derinbay 
// 150180705
// on 3/17/19.
//
#ifndef OOP_HW1_POLYNOMIAL_H
#define OOP_HW1_POLYNOMIAL_H


class Polynomial {
    int degree;
    int* coef_array;
    int polynomial_no;
    static int polynomials_created;
public:
    // Default constructor
    Polynomial();
    // Constructor
    Polynomial(const int,const int*);
    // Copy constructor
    Polynomial(const Polynomial&);
    // Operator overload +
    Polynomial operator+(const Polynomial&) const;
    // Operator overload *
    Polynomial operator*(const Polynomial&) const;
    // Temporary print
    const void print();
    // Clear polynomial counter
    static void clear_counter(){polynomials_created = 0;};
    ~Polynomial();
};


#endif //OOP_HW1_POLYNOMIAL_H
