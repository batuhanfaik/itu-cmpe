/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * /
* @ Filename: main
* @ Date: 31-Mar-2021
* @ AUTHOR: batuhanfaik
* @ Copyright (C) 2021 Batuhan Faik Derinbay
* @ Project: hw1
* @ Description: Cryptarithmetic, to run: ./hw1 DFS TWO TWO FOUR outputFileName
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <iostream>
#include <string>
#include <vector>
#include <chrono>   // Required to measure time

#include "tree.h"

using namespace std;

int main(int argc, char **argv) {
  // Get command line arguments
  string algorithm, operand1, operand2, result, output_file_name;
  if (argc == 6) {
    algorithm = argv[1];
    operand1 = argv[2];
    operand2 = argv[3];
    result = argv[4];
    output_file_name = argv[5];
  } else {
    cout << "5 arguments were expected, but got " << argc << ".\nExiting!" << endl;
    exit(1);
  }

  // Get the current time
  auto start_time = chrono::high_resolution_clock::now();

  // Create the tree
  Tree *cryptarithmetic_tree = new Tree(operand1, operand2, result);

  // Searching ends
  auto stop_time = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::microseconds>(stop_time - start_time);
  cout << "* Elapsed time of execution: " << duration.count() << " microseconds" << endl;

  return 0;
}