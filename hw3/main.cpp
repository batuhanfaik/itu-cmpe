/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * /
* @ Filename: main
* @ Date: 04-Jan-2021
* @ AUTHOR: batuhanfaik
* @ Copyright (C) 2020 Batuhan Faik Derinbay
* @ Project: hw2
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
    file.close();

    return 0;
}
