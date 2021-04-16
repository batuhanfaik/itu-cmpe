/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * /
* @ Filename: q1
* @ Date: 16-Apr-2021
* @ AUTHOR: batuhanfaik
* @ Copyright (C) 2021 Batuhan Faik Derinbay
* @ Project: hw2
* @ Description: Kruskal's Algorithm for HW2
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <iostream>
#include <fstream>

using namespace std;

int main() {
  ifstream q1File("city_plan_1.txt");    // input.txt has integers, one per line
  if (!q1File) {
    cerr << "File cannot be opened!";
    exit(1);
  }

  string v1, v2, linebreak;
  int weight;
  while (!q1File.eof()) {
    getline(q1File, v1, ',');
    getline(q1File, v2, ',');
    q1File >> weight;
    getline(q1File, linebreak, '\n'); //this is for reading the \n character into dummy variable.
    cout << v1 << " " << v2 << " " << weight << endl;
  }

  q1File.close();
  return 0;
}