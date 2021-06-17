/*
* q3_maxProfit_skeleton.cpp
*
* Created on: June 14th, 2021
*     Author: Uður Önal
*/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <set>
#include <utility>
#include <vector>
#include <algorithm>

using namespace std;

pair<int, set<int>> MaxProfit(int numOfCrystals, vector<int> profits, vector<int> entryCosts)
{
	/****************************************************************/
	/********* YOU NEED TO USE HERE AS BASE OF YOUR METHOD! *********/
	/* YOU CAN USE HELPER METHODS BUT main() WILL CALL THIS METHOD! */
	/****************************************************************/

	cout << "Dynaming Programming Table" << endl;
	for (int i = 0; i <= numOfCities; i++) {
		for (int j = 0; j <= numOfCrystals; j++) {
			cout << std::right << std::setw(3) << matrix[i][j];
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

