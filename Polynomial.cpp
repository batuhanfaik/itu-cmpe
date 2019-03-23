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
Polynomial::Polynomial(int degree, int * coef_arr):degree(degree) {
    // Polynomial counter
    polynomial_no = ++polynomials_created;

    // Allocate memory for the coefficient array
    coef_array = new int[degree];

    // Copy values of coefficients
    for (int i = 0; i < degree; ++i) {
        coef_array[i] = coef_arr[i];
    }
}

// Temporary print function
const void Polynomial::print() {
    cout << "Polynomial# " << polynomial_no << endl;
    for (int i = 1; i < degree; ++i) {
        if(coef_array[i] != 0) {
            if (coef_array[i] == 1) { cout << "x^" << (degree - i) << " + "; }
            else { cout << coef_array[i] << "x^" << (degree - i) << " + "; }
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