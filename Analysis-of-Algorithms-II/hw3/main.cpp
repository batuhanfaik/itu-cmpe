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
#define MISMATCH -5
#define GAP -10

using namespace std;

int score(const char &, const char &);
vector<vector<int>> similarity_matrix(const string &, const string &);
vector<pair<int, int>> max_similarity_indices(const vector<vector<int>> &);
string traceback_iterative(const vector<vector<int>> &, int, int, string);
string traceback_recursive(const vector<vector<int>> &, int, int, string, string);
void print_similarity_matrix(const vector<vector<int>> &);

int main() {
  ifstream infile;
  infile.open("strings.txt");

  if (!infile) {
    cerr << "File cannot be opened!";
    exit(1);
  }

  vector<string> lines;
  string line;
  // Read first N lines
  while (!infile.eof()) {
    // Read strings from infile
    getline(infile, line, '\n'); //line (string)
    line.erase(remove(line.begin(), line.end(), '\r'), line.end());
    lines.push_back(line); // Add line to the lines
  }

  // Close the infile
  infile.close();
  // Open the output file
  ofstream outfile;
  outfile.open("output.txt");
  // Sort lines read from the infile
  sort(lines.begin(), lines.end());

  // Compare strings to one another
  for (uint wi = 0; wi < lines.size(); wi++) {
    for (uint wj = wi + 1; wj < lines.size(); wj++) {
      outfile << lines[wi];
      outfile << " - " << lines[wj] << endl;
      vector<vector<int>> sm = similarity_matrix(lines[wi], lines[wj]);
//      print_similarity_matrix(sm);
      vector<string> sequences;
      for (auto const &i : max_similarity_indices(sm)) {
        sequences.push_back(traceback_iterative(sm, i.first, i.second, lines[wi]));
      }

      if (sequences.empty()) {
        outfile << "Score: 0 Sequence(s):" << endl;
      } else {
        // Sort sequences
        sort(sequences.begin(), sequences.end());
        // Delete duplicate sequences
        sequences.erase(unique(sequences.begin(), sequences.end()), sequences.end());
        outfile << "Score: " << sequences[0].length() << " Sequence(s):";
        for (auto const &i : sequences) {
          outfile << " \"" << i << "\"";
        }
        outfile << endl;
      }
    }
  }
  // Close the outfile
  outfile.close();
}

int score(const char &c1, const char &c2) {
  if (c1 != c2)
    return MISMATCH;
  return MATCH;
}

vector<vector<int>> similarity_matrix(const string &w1, const string &w2) {
  string word1 = "-" + w1;
  string word2 = "-" + w2;
  vector<vector<int>> matrix(word1.length(), vector<int>(word2.length(), 0));

  // Fill the matrix
  for (uint i = 1; i < word1.length(); i++) {
    for (uint j = 1; j < word2.length(); j++) {
      matrix[i][j] = max({
                             matrix[i - 1][j - 1] + score(word1[i], word2[j]),
                             matrix[i - 1][j] + GAP,
                             matrix[i][j - 1] + GAP,
                             0});
    }
  }

  return matrix;
}

vector<pair<int, int>> max_similarity_indices(const vector<vector<int>> &sm) {
  vector<pair<int, int>> max_indices;
  int max = MATCH;

  for (uint i = 1; i < sm.size(); i++) {
    for (uint j = 1; j < sm[i].size(); j++) {
      if (sm[i][j] > max) {
        max = sm[i][j];
        max_indices.clear();
        max_indices.push_back(make_pair(i, j));
      } else if (sm[i][j] == max) {
        max_indices.push_back(make_pair(i, j));
      }
    }
  }

  return max_indices;
}

string traceback_iterative(const vector<vector<int>> &sm, int i, int j, string w1) {
  string substr = "";
  while (sm[i][j]) {
    substr = w1[i - 1] + substr;
    i--;
    j--;
  }
  return substr;
}

string traceback_recursive(const vector<vector<int>> &sm, int i, int j, string acc, string w1) {
  if (!sm[i][j])
    return acc;
  acc = w1[i - 1] + acc;
  return traceback_recursive(sm, i - 1, j - 1, acc, w1);
}

void print_similarity_matrix(const vector<vector<int>> &sm) {
  for (const auto &i: sm) {
    for (const auto &j: i) {
      cout << j << "\t";
    }
    cout << endl;
  }
}