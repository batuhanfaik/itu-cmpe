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
const void print_help();
const void list_polynomials(int, const Polynomial*);
const void list_vectors(int, const Vector*);
const Polynomial polynomial_op(const string&, int, const Polynomial*);
const Vector vector_op(const string&, int, const Vector*);

int main(){
    int operation_selection = 4;
    int polynomial_amount = 0, vector_amount = 0;
    // Pointers of polynomials and vectors
    Polynomial* polynomials;
    Vector* vectors;

    // Read polynomials
    tie(polynomial_amount, polynomials) = read_polynomials();
    // Read vectors
    tie(vector_amount, vectors) = read_vectors();

    // Run the program as long as the user doesn't end it
    while (operation_selection != 0){
        switch(operation_selection) {
            case 1: {
                // List polynomials
                list_polynomials(polynomial_amount, polynomials);
                // List vectors
                list_vectors(vector_amount, vectors);
                break;
            }
            case 2: {
                // Read user input
                string user_op;
                cout << "+: Polynomial addition" << endl
                << "*: Polynomial multiplication" << endl
                << "(Operations should be entered without spaces)" << endl
                << "Enter operation: ";
                cin >> user_op;
                // Do a poly op
                Polynomial result = polynomial_op(user_op, polynomial_amount, polynomials);
                if (result.getDegree() != 0){
                    cout << "Result: " << result << endl;
                }
                break;
            }
            case 3: {
                // Read user input
                string user_op;
                cout << "+: Vector addition" << endl
                << "*: Scalar multiplication" << endl
                << ".: Dot product" << endl
                << "(Operations should be entered without spaces)" << endl
                << "Enter operation: ";
                cin >> user_op;
                // Do a vector op
                Vector result = vector_op(user_op, vector_amount, vectors);
                // Print only if returned value is non-empty object
                if (result.getSize() != 0){
                    cout << "Result: " << result << endl;
                }
                break;
            }
            case 4: {
                // Print help
                print_help();
                break;
            }
            default: // Input not recognized
                cout << "Input not recognized." << endl
                << "Please enter a valid action." << endl;
                break;
        }
        // Read the operation selector
        cout << "Enter an option: ";
        cin >> operation_selection;
    }

    cout << "Exiting the program..." << endl;
    return 0;
}

tuple <int,Polynomial*> read_polynomials(){
    // Stores lines
    string line;

    // Start read stream and open file
    ifstream polynomial_file;
    polynomial_file.open("Polynomial.txt");

    if(!polynomial_file.is_open()){
        cout << "Error opening the polynomial file" << endl;
        // Return empty tuple
        return make_tuple(0, nullptr);
    } else{
        // Find the amount of polynomials and store
        getline(polynomial_file, line);
        // string is casted to int
        int polynomial_amount = stoi(line);

        // Create Vector object array
        Polynomial* polynomial_array;
        // Allocate memory to the vector array
        polynomial_array = new Polynomial[polynomial_amount];

        // Reset vector counter
//        Polynomial::clear_counter();

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
                int coefficient_amount = stoi(coefficient_str);
                // Coefficient amount is incremented because we read the degree from the file
                // Coefficient amount is always one higher than the degree
                coefficient_amount++;

                // Create the array to store coefficients of the polynomial
                float* coefficients_array;
                // Reserve memory for the value array
                coefficients_array = new float[coefficient_amount];

                // Go through all coefficients and store them in the vector value array
                for (int coefficient_index = 0; coefficient_index < coefficient_amount; ++coefficient_index){
                    // Read a coefficient
                    ss >> coefficient_str;
                    // Append it to the value array
                    coefficients_array[coefficient_index] = stoi(coefficient_str);
                }
                // Required for getting rid of end of the line
                ss >> coefficient_str;

                // Create a Polynomial object with the given values
                polynomial_array[i] = Polynomial(coefficient_amount, coefficients_array);

                // Release the memory for coefficients array
                delete[] coefficients_array;
            }
        }
        // Polynomials are read successfully
        cout << "Polynomials are read successfully!" << endl;
        // Return the polynomial object array with its size
        return make_tuple(polynomial_amount, polynomial_array);
    }
}
tuple <int,Vector*> read_vectors(){
    // Stores lines
    string line;

    // Start read stream and open file
    ifstream vector_file;
    vector_file.open("Vector.txt");

    if(!vector_file.is_open()){
        cout << "Error opening the vector file" << endl;
        // Return empty tuple
        return make_tuple(0, nullptr);
    } else{
        // Find the amount of vectors and store
        getline(vector_file, line);
        // string is casted to int
        int vector_amount = stoi(line);

        // Create Vector object array
        Vector* vector_array;
        // Allocate memory to the vector array
        vector_array = new Vector[vector_amount];

        // Reset vector counter
//        Vector::clear_counter();

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
                int coefficient_amount = stoi(coefficient_str);

                // Create the array to store values of the vector
                float* vector_values;
                // Reserve memory for the value array
                vector_values = new float[coefficient_amount];

                // Go through all coefficients and store them in the vector value array
                 for (int coefficient_index = 0; coefficient_index < coefficient_amount; ++coefficient_index){
                    // Read a coefficient
                    ss >> coefficient_str;
                    // Append it to the value array
                    vector_values[coefficient_index] = stoi(coefficient_str);
                }
                 // Required for getting rid of end of the line
                 ss >> coefficient_str;

                // Create a Vector object with the given values
                vector_array[i] = Vector(coefficient_amount, vector_values);

                // Release the memory for vector value array
                delete[] vector_values;
             }
         }
        // Vectors are read successfully
        cout << "Vectors are read successfully!" << endl;
        // Return the vector object array with its size
        return make_tuple(vector_amount, vector_array);
    }
}
const void print_help(){
    cout << endl << "Possible Actions:" << endl
    << "1. Print Polynomial and Vector lists" << endl
    << "2. Do a polynomial operation" << endl
    << "3. Do a vector operation" << endl
    << "4. Help: Print possible actions" << endl
    << "0. Exit the program" << endl << endl;
}
const void list_polynomials(int polynomial_amount, const Polynomial* const polynomials){
    cout << "Polynomials:" << endl;

    // Print polynomials
    for(int i=0; i < polynomial_amount; ++i){
        cout << (i+1) << ". ";
        polynomials[i].print();
    }
    cout << endl;
}
const void list_vectors(int vector_amount, const Vector* const vectors){
    cout << "Vectors:" << endl;

    // Print polynomials
    for(int i=0; i < vector_amount; ++i){
        cout << (i+1) << ". ";
        vectors[i].print();
    }
    cout << endl;
}
const Polynomial polynomial_op(const string& user_op, int polynomial_amount, const Polynomial* polynomials){
    // Find which operator is used
    size_t add_op_found = user_op.find("+");
    size_t mult_op_found = user_op.find("*");

    // Do the operation accordingly
    // If the operation is addition
    if(add_op_found != string::npos){
        // Get the index of the first polynomial expression
        int first_poly = stoi(user_op.substr(0,add_op_found));
        // Get the index of the second polynomial expression
        int second_poly = stoi(user_op.substr(add_op_found + 1, user_op.length()));
        // Check if indexes are valid
        if (first_poly < polynomial_amount && second_poly < polynomial_amount){
           // Decrement indexes by one (arrays start at 0 :) )
            first_poly--;
            second_poly--;
            // Return the addition
            return (polynomials[first_poly]+polynomials[second_poly]);
        } else { // If indexes are not valid
            cout << "Indexes are not in range." << endl;
            return Polynomial(0, nullptr);
        }
    // If the operation is multiplication
    } else if(mult_op_found != string::npos){
        // Get the index of the first polynomial expression
        int first_poly = stoi(user_op.substr(0,mult_op_found));
        // Get the index of the second polynomial expression
        int second_poly = stoi(user_op.substr(mult_op_found + 1, user_op.length()));
        // Check if indexes are valid
        if (first_poly < polynomial_amount && second_poly < polynomial_amount){
            // Decrement indexes by one (arrays start at 0 :) )
            first_poly--;
            second_poly--;
            // Return the multiplication
            return (polynomials[first_poly]*polynomials[second_poly]);
        } else { // If indexes are not valid
            cout << "Indexes are not in range." << endl;
            return Polynomial(0, nullptr);
        }
    } else {
        cout << "Unknown operator." << endl;
        return Polynomial(0, nullptr);
    }
}
const Vector vector_op(const string& user_op, int vector_amount, const Vector* vectors){
    // Find which operator is used
    size_t add_op_found = user_op.find("+");
    size_t mult_op_found = user_op.find("*");
    size_t dot_op_found = user_op.find(".");

    // Do the operation accordingly
    // If the operation is addition
    if(add_op_found != string::npos){
        // Get the index of the first vector
        int first_vector = stoi(user_op.substr(0,add_op_found));
        // Get the index of the second vector
        int second_vector = stoi(user_op.substr(add_op_found + 1, user_op.length()));
        // Check if indexes are valid
        if (first_vector < vector_amount && second_vector < vector_amount){
           // Decrement indexes by one (arrays start at 0 :) )
            first_vector--;
            second_vector--;
            // Return the addition
            return (vectors[first_vector]+vectors[second_vector]);
        } else { // If indexes are not valid
            cout << "Indexes are not in range." << endl;
            return Vector(0, nullptr);
        }
    // If the operation is multiplication
    } else if(mult_op_found != string::npos){
        // Get the index of the vector
        int vector = stoi(user_op.substr(0,mult_op_found));
        // Get the scalar number
        float scalar = stoi(user_op.substr(mult_op_found + 1, user_op.length()));
        // Check if the index is valid
        if (vector < vector_amount){
            // Decrement the index by one (arrays start at 0 :) )
            vector--;
            // Return the multiplication
            return (vectors[vector]*scalar);
        } else { // If indexes are not valid
            cout << "Indexes are not in range." << endl;
            return Vector(0, nullptr);
        }
    // If the operation is dot product
    } else if(dot_op_found != string::npos){
        // Get the index of the first vector
        int first_vector = stoi(user_op.substr(0,dot_op_found));
        // Get the index of the second vector
        int second_vector = stoi(user_op.substr(dot_op_found + 1, user_op.length()));
        // Check if indexes are valid
        if (first_vector < vector_amount && second_vector < vector_amount){
            // Decrement indexes by one (arrays start at 0 :) )
            first_vector--;
            second_vector--;
            float dot_product = vectors[first_vector] * vectors[second_vector];
            cout << "Result: " << dot_product << endl << endl;
        } else { // If indexes are not valid
            cout << "Indexes are not in range." << endl;
        }
        return Vector(0, nullptr);
    } else {
        cout << "Unknown operator." << endl;
        return Vector(0, nullptr);
    }
}
