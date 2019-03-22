//
// Created by;
// Batuhan Faik Derinbay 
// 150180705
// on 3/17/19.
//

#include <iostream>
#include <fstream>
#include <list>
#include <string>
#include <sstream>
#include <array>
#include <tuple>
#include "Vector.h"
#include "Polynomial.h"

using namespace std;

// tie, tuple and make_tuple: https://stackoverflow.com/questions/35098211/how-do-i-return-two-values-from-a-function
tuple <int,Vector*> read_vectors();
void read_polynomials();

int main(){
    int vector_amount = 0;
    Vector* vectors;

    // Read vectors
    tie(vector_amount, vectors) = read_vectors();
    // read_polynomials();

    // Print vectors
    for(int i=0; i < vector_amount; ++i){
        vectors[i].print();
    }

    return 0;
}

tuple <int,Vector*> read_vectors(){
    // Stores lines
    string line;

    // Start read stream and open file
    ifstream vector_file;
    vector_file.open("../Vector.txt");

    if(!vector_file.is_open()){
        cout << "Error opening the file" << endl;
    } else{
        // Find the amount of vectors and store
        getline(vector_file, line);
        // string is casted to int
        int vector_amount = std::stoi(line);

        // Create Vector object array
        Vector* vector_array;
        // Allocate memory to the vector array
        vector_array = new Vector[vector_amount];

        // Reset vector counter
        Vector::clear_counter();

        for (int i = 0; i < vector_amount; ++i) {        // As long as the line is readable
             if (vector_file.good()) {
                 getline(vector_file, line);         // Read the i'th line

                 // Used to split string around spaces.
                istringstream ss(line);
                 // Reads the amount of coefficients
                string coefficient_str;
                ss >> coefficient_str;
                // Store the amount of coefficients
                int coefficient_amount = std::stoi(coefficient_str);

                // Create the array to store values of the vector
                int* vector_values;
                // Reserve memory for the value array
                vector_values = new int[coefficient_amount];

                // Go through all coefficients and store them in the vector value array
                 for (int coefficient_index = 0; coefficient_index < coefficient_amount; ++coefficient_index){
                    // Read a coefficient
                    ss >> coefficient_str;
                    // Append it to the value array
                    vector_values[coefficient_index] = std::stoi(coefficient_str);
                }
                 // Required for getting rid of end of the line
                 ss >> coefficient_str;

                // Create a Vector object with the given values
                vector_array[i] = Vector(coefficient_amount, vector_values);

                // Release the memory for vector value array
                delete[] vector_values;
             }
         }
        // Return the vector object array
        return make_tuple(vector_amount, vector_array);
    }
}