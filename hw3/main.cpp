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
#include <algorithm>

#define MATCH 1
#define MISMATCH -1
#define GAP -2

using namespace std;

int score(const char&, const char&);
vector<vector<int>> similarity_matrix(const string&, const string&);

int main() {
  ifstream file;
  file.open("strings.txt");

  if (!file) {
    cerr << "File cannot be opened!";
    exit(1);
  }

  vector<string> lines;
  string line;
  // Read first N lines
//  while (!file.eof()) {
//    // Read strings from file
//    getline(file, line, '\n'); //line (string)
//    lines.push_back(line); // Add line to the lines
//  }

  vector<vector<int>> sm = similarity_matrix("information", "funimatio");
  for (const auto& i: sm) {
    for (const auto& j: i) {
      cout << j << "\t";
    }
    cout << endl;
  }

  // Print lines
//  for (const auto& e: lines) {
//    cout << e << endl;
//  }

  file.close();
}

int score(const char& c1, const char& c2){
  if (c1 != c2)
    return MISMATCH;
  return MATCH;
}

vector<vector<int>> similarity_matrix(const string& w1, const string& w2) {
  string word1 = "-" + w1;
  string word2 = "-" + w2;
  vector<vector<int>> matrix(word1.length(), vector<int> (word2.length(), 0));

  // Fill the matrix
  for (uint i=1; i < word1.length(); i++) {
    for (uint j=1; j < word2.length(); j++) {
      matrix[i][j] = max({
          matrix[i-1][j-1] + score(word1[i], word2[j]),
          matrix[i-1][j] + GAP,
          matrix[i][j-1] + GAP,
          0});
    }
  }

  return matrix;
}