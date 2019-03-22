//
// Created by;
// Batuhan Faik Derinbay 
// 150180705
// on 3/17/19.
//
#include "Vector.h"
#include <iostream>
using namespace std;

// Set created vector amount to 0
int Vector::vectors_created;

// Default constructor
Vector::Vector():size(0),value_array(nullptr){
    // Vector counter
    vector_no = ++vectors_created;
}

// Constructor
Vector::Vector(int size, int * value_arr):size(size) {
    // Vector counter
    vector_no = ++vectors_created;

    // Allocate memory
    value_array = new int[size];

    // Copy values of the array
    for (int i = 0; i < size; ++i) {
        value_array[i] = value_arr[i];
    }
}

// Temporary print function
const void Vector::print() {
    cout << "Vector# " << vector_no << endl << "( ";
    for (int i = 0; i < size; ++i) {
        cout << value_array[i] << " ";
    }
    cout << ")" << endl;
}

// Destructor
Vector::~Vector(){
    --vectors_created;
//    delete[] value_array;
}