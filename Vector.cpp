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
Vector::Vector(const int size,const int * value_arr):size(size) {
    // Vector counter
    vector_no = ++vectors_created;

    // Allocate memory
    value_array = new int[size];

    // Copy values of the array
    for (int i = 0; i < size; ++i) {
        value_array[i] = value_arr[i];
    }
}

// Copy constructor
Vector::Vector(const Vector& vector_in) {
    size = vector_in.size;
    // Allocate memory for the new array
    value_array = new int[size];
    // Copy the values over
    for (int i = 0; i < size; ++i) {
        value_array[i] = vector_in.value_array[i];
    }
}

// Getter methods
int Vector::getSize() const{
    return size;
}
int Vector::getValueArray(int i) const{
    return value_array[i];
}
int Vector::getVectorNo() const{
    return vector_no;
}

// Operator overload +
Vector Vector::operator+(const Vector& vector_in) const{
    // If sizes match
    if (getSize() == vector_in.getSize()){
        int* new_value_array;
        new_value_array = new int[getSize()];
        // Do the addition
        for (int i = 0; i < getSize(); ++i) {
            new_value_array[i] = getValueArray(i) + vector_in.getValueArray(i);
        }
        // Return the resulting vector
        return Vector(getSize(), new_value_array);
    // If sizes don't match
    } else {
        cout << "Vector sizes don't match" << endl;
        // Return an empty vector
        return Vector(0,nullptr);
    }
}

// Operator overload * (Dot product)
int Vector::operator*(const Vector& vector_in) const{
    // If sizes match
    if (getSize() == vector_in.getSize()){
        int* new_value_array;
        new_value_array = new int[getSize()];
        // Do the addition
        int sum = 0;
        for (int i = 0; i < getSize(); ++i) {
            new_value_array[i] = getValueArray(i) * vector_in.getValueArray(i);
            sum += new_value_array[i];
        }
        // Return the resulting vector
        return sum;
    // If sizes don't match
    } else {
        cout << "Vector sizes don't match" << endl;
        // Return 0
        return 0;
    }
}

// Operator overload * (Scalar multiplication)
Vector Vector::operator*(const int scalar) const{
    // Create a new value array
    int* new_value_array;
    new_value_array = new int[getSize()];
    // Multiply values by the scalar
    for (int i = 0; i < getSize(); ++i) {
        new_value_array[i] = getValueArray(i)*scalar;
    }
    // Return the new vector
    return Vector(getSize(), new_value_array);
}

// Temporary print function
const void Vector::print() {
//    cout << "Vector# " << vector_no << endl << "(";
    cout << "(";
    for (int i = 0; i < (size - 1); ++i) {
        cout << value_array[i] << ", ";
    }
    cout << value_array[size - 1] << ")" << endl;
}

// Destructor
Vector::~Vector(){
//    --vectors_created;
//    delete[] value_array;
}