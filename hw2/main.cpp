/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * /
* @ Filename: main
* @ Date: 20-Dec-2020
* @ AUTHOR: batuhanfaik
* @ Copyright (C) 2020 Batuhan Faik Derinbay
* @ Project: hw2
* @ Description: Read from locations (driver code)
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>   // Required to measure time
#include <random>   // Required to generate random numbers

#include "taxi.h"
#include "heap.h"

using namespace std;

template<typename T>
T get_random(T range_from, T range_to) {
    random_device rand_dev;
    mt19937 generator(rand_dev());
    uniform_real_distribution<T> distribution(range_from, range_to);
    return distribution(generator);
}

int main(int argc, char** argv) {
    int m = 1000;
    double p = 0.2;
    if (argc < 3){
        cout << "m and p values need to be passed in.\nThis run will assume that m=1000, p=0.2" << endl;
    } else if (argc > 3){
        cout << "More than two parameters are passed in.\nThis run will assume that m=1000, p=0.2" << endl;
    } else {
        m = stoi(argv[1]);
        p = stof(argv[2]);
    }

    ifstream file;
    file.open("locations.txt");

    if (!file) {
        cerr << "File cannot be opened!";
        exit(1);
    }

    double hotel_long = 33.40819;
    double hotel_lat = 39.19001;

    string line;
    int taxi_additions = 0, distance_updates = 0;
    double taxi_long, taxi_lat, rand_number;
    getline(file, line); // this is the header line

    // Get the current time
    auto start_time = chrono::high_resolution_clock::now();
    // Start simulation
    for (int i = 0; i < m; i++) {
        rand_number = get_random(double(0), double(1));
        if (rand_number <= p){    // Update taxi (decrease the distance)
            Heap::update_random_taxi();
            distance_updates++;
        } else {    // Read a new taxi object (add to heap)
            file >> taxi_long; // longitude of the taxi (float)
            file >> taxi_lat; // latitude of the taxi (float)
            getline(file, line, '\n'); // this is for reading the \n character into dummy variable.
            Heap::add_taxi(Taxi(taxi_long, taxi_lat, hotel_long, hotel_lat));
            taxi_additions++;
        }
        if (i % 99 == 0)    // Call a taxi (remove from heap)
            Heap::call_taxi();
    }
    // Simulation ends, so print execution time
    auto stop_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop_time - start_time);
    cout << "For m=" << m << " and p=" << p << endl << "Elapsed time of execution: " << duration.count()
         << " microseconds" << endl;

    return 0;
}