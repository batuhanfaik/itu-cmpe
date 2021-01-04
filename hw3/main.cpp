/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * /
* @ Filename: main
* @ Date: 04-Jan-2021
* @ AUTHOR: batuhanfaik
* @ Copyright (C) 2020 Batuhan Faik Derinbay
* @ Project: hw3
* @ Description: Read from locations (driver code)
* @ Compiling: g++ -o 150180705.out main.cpp ----
* @ Running: ./150180705.out <csv_file> (E.g., ./150180705.out sample.csv)
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

int main(int argc, char** argv) {
    string filename = "sample.csv";
    if (argc < 2){
        cout << "An CSV filename needs to be passed in.\nThis run will use \"sample.csv\"" << endl;
    } else if (argc > 2){
        cout << "More than one parameters are passed in.\nThis run will use \"sample.csv\"" << endl;
    } else {
        filename = argv[1];
    }

    ifstream file;
    file.open(filename);

    if (!file) {
        cerr << "File cannot be opened!";
        exit(1);
    }

    string line, season, full_name, team;
    int rebound, assist, point;

    getline(file, line); // Read the header line

    // Read until the end of the file
    while (!file.eof()){
        // Read values of the Players
        getline(file, season, ',');     // Read season
        getline(file, full_name, ',');    // Read full name
        getline(file, team, ',');    // Read team
        getline(file, line, ',');    // Read rebound
        rebound = stoi(line);
        getline(file, line, ',');    // Read assist
        assist = stoi(line);
        getline(file, line, ',');    // Read point
        point = stoi(line);
        getline(file, line, '\n');    // Read the last new line character
    }

    file.close();

    return 0;
}
