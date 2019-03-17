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

using namespace std;

const int MAX_COEF_AMOUNT = 256;

void read_vectors();
void read_polynomials();
//int *parse_line(string, int []);

int main(){
    read_vectors();
//    read_polynomials();
    return 0;
}

void read_vectors(){
    string line;

    ifstream vector_file;
    vector_file.open("../Vector.txt");

    if(!vector_file.is_open()){
        cout << "Error opening the file" << endl;
    } else{
        //Find the amount of vectors and store
        getline(vector_file, line);
        //string is casted to int
        int vector_amount = std::stoi(line);
        int max_coef_amount = 0;
        int coefficient_matrix[vector_amount][MAX_COEF_AMOUNT];

        for (int i = 0; i < vector_amount; ++i) {        //As long as the line is readable
             if (vector_file.good()) {
                 getline(vector_file, line);         //Read the i'th line
//                 cout << line << endl;

                 // Used to split string around spaces.
                istringstream ss(line);
                 //Reads the amount of coefficients
                string coefficient_str;
                ss >> coefficient_str;
                int coefficient_amount = std::stoi(coefficient_str);

                if (coefficient_amount > max_coef_amount) max_coef_amount = coefficient_amount;

                int coefficient_index = 0;

                // Traverse through all coefficients while there is more to read
                while(ss) {
                    // Read a coefficient
                    ss >> coefficient_str;
                    // Append it to the coefficient matrix
                    coefficient_matrix[i][coefficient_index] = std::stoi(coefficient_str);
                    // Increase the index by one
                    coefficient_index++;
                }
             }
         }
        for (int i = 0; i < vector_amount; ++i) {
            for (int j = 0; j < max_coef_amount; ++j) {
                cout << coefficient_matrix[i][j] << " ";
            }
            cout << "\n";
        }
    }
}