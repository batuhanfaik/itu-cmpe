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

using namespace std;

int main(int argc, char **argv) {
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

  cout << algorithm << operand1 << operand2 << result << output_file_name << endl;

  return 0;
}