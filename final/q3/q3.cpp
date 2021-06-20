/*
* q3_maxProfit_skeleton.cpp
*
* Created on: June 14th, 2021
*     Author: Ugur Unal
*/
/***********************************************************
STUDENT INFORMATION
Full Name : Batuhan Faik Derinbay
Student ID:  150180705
**********************************************************/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <set>
#include <utility>
#include <vector>
#include <algorithm>

using namespace std;

pair<int, set<int>> MaxProfit(int numOfCrystals, vector<int> profits, vector<int> entryCosts) {
  /****************************************************************/
  /********* YOU NEED TO USE HERE AS BASE OF YOUR METHOD! *********/
  /* YOU CAN USE HELPER METHODS BUT main() WILL CALL THIS METHOD! */
  /****************************************************************/
  int N = int(profits.size()), W = numOfCrystals;
  // Fill 'matrix' with zeros
  vector<vector<int>> matrix = vector<vector<int>>((N + 1), vector<int>((W + 1), 0));

  for (int i = 1; i <= N; i++) {
    for (int w = 1; w <= W; w++) {
      if (entryCosts[i - 1] > w)
        matrix[i][w] = matrix[i - 1][w];
      else
        matrix[i][w] = max(matrix[i - 1][w], profits[i - 1] + matrix[i - 1][w - entryCosts[i - 1]]);
    }
  }

  set<int> citiesToVisit;
  int maxProfit = matrix[N][W];
  int i = N, j = W, numOfCities = int(matrix.size()) - 1;

  while (i > 0 && j > 0) {
    if (matrix[i][j] != matrix[i - 1][j]) {
      citiesToVisit.insert(i);
      j -= entryCosts[i - 1];
    }
    i--;
  }

  cout << "Dynamic Programming Table" << endl;
  for (i = 0; i <= numOfCities; i++) {
    for (j = 0; j <= numOfCrystals; j++) {
      cout << right << setw(3) << matrix[i][j];
    }
    cout << endl;
  }

  return pair<int, set<int>>(maxProfit, citiesToVisit);
}

int main() {
  int numOfCrystals;
  vector<int> profits;
  vector<int> entryCosts;

  string inputFilename;
  cout << "Enter the name of the input file: ";
  cin >> inputFilename;

  ifstream input(inputFilename);

  if (!input.is_open()) {
    cerr << "File named \"" << inputFilename << "\" could not open!" << endl;
    return EXIT_FAILURE;
  }

  string line;
  if (getline(input, line)) {
    numOfCrystals = stoi(line);
  }
  while (getline(input, line, ' ')) {
    profits.push_back(stoi(line));
    getline(input, line);
    entryCosts.push_back(stoi(line));
  }

  pair<int, set<int>> result = MaxProfit(numOfCrystals, profits, entryCosts);

  cout << "Max profit is " << result.first << "." << endl;
  cout << "Cities visited:";
  for (int cityNumber : result.second) {
    cout << " " << cityNumber;
  }
  cout << endl;
}

