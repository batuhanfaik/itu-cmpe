//
// Created by;
// Batuhan Faik Derinbay 
// 150180705
// on 3/17/19.
//
#include "Polynomial.h"
#include <iostream>
using namespace std;

// Set created vector amount to 0
int Polynomial::polynomials_created;

// Default constructor
Polynomial::Polynomial():degree(0),coef_array(nullptr){
    // Polynomial counter
    polynomial_no = ++polynomials_created;
}

// Constructor
// FOR SIMPLIFICATION, DEGREE DENOTES THE NUMBER OF COEFFICIENTS
// THAT THE PARTICULAR POLYNOMIAL CAN HAVE, NOT THE ACTUAL DEGREE
Polynomial::Polynomial(const int degree,const int * coef_arr):degree(degree) {
    // Polynomial counter
    polynomial_no = ++polynomials_created;

    // Allocate memory for the coefficient array
    coef_array = new int[degree];

    // Copy values of coefficients
    for (int i = 0; i < degree; ++i) {
        coef_array[i] = coef_arr[i];
    }
}

// Copy constructor
Polynomial::Polynomial(const Polynomial& polynomial_in) {
    degree = polynomial_in.degree;
    // Allocate memory for the new array
    coef_array = new int[degree];
    // Copy the values over
    for (int i = 0; i < degree; ++i) {
        coef_array[i] = polynomial_in.coef_array[i];
    }
}

// Operator overload +
Polynomial Polynomial::operator+(const Polynomial& polynomial_in) const{
    int new_degree = 0;
    // Create a new coefficient array
    int* new_coef_array;

    // Assign values of the new array
    // If the left value has higher degree
    if (degree >= polynomial_in.degree){
        new_degree = degree;
        new_coef_array = new int[new_degree];
        // Assign the larger degree coefficients first
        for (int i = 0; i < (new_degree - polynomial_in.degree); ++i) {
            new_coef_array[i] = coef_array[i];
        }
        // Then assign the remaining coefficients
        for (int i = (polynomial_in.degree - 1); i < degree; ++i) {
            new_coef_array[i] = coef_array[i] + polynomial_in.coef_array[i - polynomial_in.degree + 1];
        }
        // If the right value has higher degree
    } else {
        new_degree = polynomial_in.degree;
        new_coef_array = new int[new_degree];
        // Assign the larger degree coefficients first
        for (int i = 0; i < (new_degree - degree); ++i) {
            new_coef_array[i] = polynomial_in.coef_array[i];
        }
        // Then assign the remaining coefficients
        for (int i = (degree - 1); i < degree; ++i) {
            new_coef_array[i] = coef_array[i - degree + 1] + polynomial_in.coef_array[i];
        }
    }

    return Polynomial(new_degree, new_coef_array);
}

// Operator overload *
Polynomial Polynomial::operator*(const Polynomial& polynomial_in) const{

}

// Temporary print function
const void Polynomial::print() {
//    cout << "Polynomial# " << polynomial_no << endl;
    for (int i = 0; i < (degree - 1); ++i) {
        if(coef_array[i] != 0) {
            if (coef_array[i] == 1) { cout << "x^" << (degree - i - 1) << " + "; }
            else { cout << coef_array[i] << "x^" << (degree - i - 1) << " + "; }
        }
    }
    // Print last coefficient if not 0
    if(coef_array[degree-1] != 0) {
        cout << coef_array[degree - 1];
    }
    cout << endl;
}

// Destructor
Polynomial::~Polynomial() {

}