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
void print_similarity_matrix(const vector<vector<int>>&);

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
  while (!file.eof()) {
    // Read strings from file
    getline(file, line, '\n'); //line (string)
    line.erase(remove(line.begin(), line.end(), '\r'), line.end());
    lines.push_back(line); // Add line to the lines
  }

for (uint wi=0; wi < lines.size(); wi++) {
  for (uint wj=wi+1; wj < lines.size(); wj++) {
    cout << lines[wi];
    cout << " vs " << lines[wj] << endl;
    vector<vector<int>> sm = similarity_matrix(lines[wi], lines[wj]);
//    print_similarity_matrix(sm);
  }
}

  // Close the file
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

void print_similarity_matrix(const vector<vector<int>>& sm) {
  for (const auto& i: sm) {
    for (const auto& j: i) {
      cout << j << "\t";
    }
    cout << endl;
  }
}