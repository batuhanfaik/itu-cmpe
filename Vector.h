//
// Created by;
// Batuhan Faik Derinbay 
// 150180705
// on 3/17/19.
//
#ifndef OOP_HW1_VECTOR_H
#define OOP_HW1_VECTOR_H


class Vector {
    int size;
    int* value_array;
    int vector_no;
    static int vectors_created;
public:
    // Default constructor
    Vector();
    // Constructor
    Vector(const int,const int*);
    // Getter methods
    int getSize() const;
    int getValueArray(int) const;
    int getVectorNo() const;
    // Operator overload +
    Vector operator+(const Vector&) const;
    // Operator overload * (Dot product)
    int operator*(const Vector&) const;
    // Operator overload * (Scalar multiplication)
    Vector operator*(const int) const;
    // Copy constructor
    Vector(const Vector&);
    // Temporary print
    const void print();
    // Clear vector counter
    static void clear_counter(){vectors_created = 0;};
    ~Vector();
};


#endif //OOP_HW1_VECTOR_H
