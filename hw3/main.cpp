/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * /
* @ Filename: main
* @ Date: 13-May-2021
* @ AUTHOR: batuhanfaik
* @ Copyright (C) 2021 Batuhan Faik Derinbay
* @ Project: hw3
* @ Description: Local Sequence Alignment with Smith-Waterman
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

int main() {
  ifstream file;
  file.open("strings.txt");

  if (!file) {
    cerr << "File cannot be opened!";
    exit(1);
  }

  string line;
  // Read first N lines
  while (!file.eof()) {
    // Read strings from file
    getline(file, line, '\n'); //line (string)
    cout << line << endl;
  }
}