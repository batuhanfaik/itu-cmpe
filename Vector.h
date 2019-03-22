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
    Vector(int, int*);
    // Temporary print
    const void print();
    // Clear vector counter
    static void clear_counter(){vectors_created = 0;};
    ~Vector();
};


#endif //OOP_HW1_VECTOR_H
