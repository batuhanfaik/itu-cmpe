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
tuple <int,Polynomial*> read_polynomials();
const void list_polynomials(const int&, Polynomial* const);
const void list_vectors(const int&, Vector* const);

int main(){
    int polynomial_amount = 0, vector_amount = 0;
    // Pointers of polynomials and vectors
    Polynomial* polynomials;
    Vector* vectors;

    // Read polynomials
    tie(polynomial_amount, polynomials) = read_polynomials();
    // Read vectors
    tie(vector_amount, vectors) = read_vectors();

    // List polynomials
    list_polynomials(polynomial_amount, polynomials);

    // List vectors
    list_vectors(vector_amount, vectors);

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

        // For every vector in the file
        for (int i = 0; i < vector_amount; ++i) {
             if (vector_file.good()) {                // As long as the line is readable
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
        // Return the vector object array with its size
        return make_tuple(vector_amount, vector_array);
    }
}

tuple <int,Polynomial*> read_polynomials(){
    // Stores lines
    string line;

    // Start read stream and open file
    ifstream polynomial_file;
    polynomial_file.open("../Polynomial.txt");

    if(!polynomial_file.is_open()){
        cout << "Error opening the file" << endl;
    } else{
        // Find the amount of polynomials and store
        getline(polynomial_file, line);
        // string is casted to int
        int polynomial_amount = std::stoi(line);

        // Create Vector object array
        Polynomial* polynomial_array;
        // Allocate memory to the vector array
        polynomial_array = new Polynomial[polynomial_amount];

        // Reset vector counter
        Polynomial::clear_counter();

        // For every polynomial in the file
        for (int i = 0; i < polynomial_amount; ++i) {
             if (polynomial_file.good()) {                // As long as the line is readable
                 getline(polynomial_file, line);          // Read the i'th line

                 // Used to split string around spaces.
                istringstream ss(line);
                 // Reads the amount of coefficients
                string coefficient_str;
                ss >> coefficient_str;
                // Store the amount of coefficients
                int coefficient_amount = std::stoi(coefficient_str);
                // Coefficient amount is incremented because we read the degree from the file
                // Coefficient amount is always one higher than the degree
                coefficient_amount++;

                // Create the array to store coefficients of the polynomial
                int* coefficients_array;
                // Reserve memory for the value array
                coefficients_array = new int[coefficient_amount];

                // Go through all coefficients and store them in the vector value array
                 for (int coefficient_index = 0; coefficient_index < coefficient_amount; ++coefficient_index){
                    // Read a coefficient
                    ss >> coefficient_str;
                    // Append it to the value array
                    coefficients_array[coefficient_index] = std::stoi(coefficient_str);
                }
                 // Required for getting rid of end of the line
                 ss >> coefficient_str;

                // Create a Vector object with the given values
                polynomial_array[i] = Polynomial(coefficient_amount, coefficients_array);

                // Release the memory for coefficients array
                delete[] coefficients_array;
             }
         }
        // Return the polynomial object array with its size
        return make_tuple(polynomial_amount, polynomial_array);
    }
}

const void list_polynomials(const int& polynomial_amount, Polynomial* const polynomials){
    cout << "Polynomials:" << endl;

    // Print polynomials
    for(int i=0; i < polynomial_amount; ++i){
        cout << (i+1) << ". ";
        polynomials[i].print();
    }
    cout << endl;
}

const void list_vectors(const int& vector_amount, Vector* const vectors){
    cout << "Vectors:" << endl;

    // Print polynomials
    for(int i=0; i < vector_amount; ++i){
        cout << (i+1) << ". ";
        vectors[i].print();
    }
    cout << endl;
}
