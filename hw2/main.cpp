/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * /
* @ Filename: main
* @ Date: 20-Dec-2020
* @ AUTHOR: batuhanfaik
* @ Copyright (C) 2020 Batuhan Faik Derinbay
* @ Project: hw2
* @ Description: Read from locations (driver code)
* @ Compiling: g++ -o a.out main.cpp heap.cpp taxi.cpp
* @ Running: ./a.out $m $p (E.g., ./a.out 1000 0.2)
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>   // Required to measure time
#include <random>   // Required to generate random numbers

#include "taxi.h"
#include "heap.h"

using namespace std;

template <typename T>
T get_random_real(T start, T end) {
    random_device rand_device;
    mt19937 generator(rand_device());
    uniform_real_distribution<T> distribution(start, end);
    return distribution(generator);
}

template <typename T>
T get_random_int(T start, T end) {
    random_device rand_device;
    mt19937 generator(rand_device());
    uniform_int_distribution<T> distribution(start, end);
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
    int taxi_additions = 0, distance_updates = 0, taxi_call = 0, empty_heap = 0;
    double taxi_long, taxi_lat, rand_real;
    Heap taxi_heap = Heap();
    getline(file, line); // this is the header line

    // Get the current time
    auto start_time = chrono::high_resolution_clock::now();
    // Start simulation
    for (int i = 0; i < m; i++) {
        rand_real = get_random_real(double(0), double(1));
        if (rand_real <= p){    // Update taxi (decrease the distance)
            if (!taxi_heap.get_size()) {    // If the heap is empty don't update taxis
                /* The reverse logic here is intentional due to compiler optimizations
                * Execution of this scope is possibly only if no operations can be made, and there has to be m number
                * of operations. So in order to satisfy the number of total operations, m is increased below.
                * By doing so the empty queue at the start problem is solved as well. */
                m++;
            } else {
                int rand_index = get_random_int(0, taxi_heap.get_size() - 1);
                taxi_heap.update_random_taxi(rand_index, 0.01);
                distance_updates++;
            }
        } else {    // Read a new taxi object (add to heap)
            file >> taxi_long; // longitude of the taxi (float)
            file >> taxi_lat; // latitude of the taxi (float)
            getline(file, line, '\n'); // this is for reading the \n character into dummy variable.
            taxi_heap.add_taxi(Taxi(taxi_long, taxi_lat, hotel_long, hotel_lat));
            taxi_additions++;
        }
        if ((i + 1) % 100 == 0) {   // Call a taxi (remove from heap)
            if(taxi_heap.call_taxi() != -1) {    // Check if the call was successful
                taxi_call++;
            }
        }
    }
    // Simulation ends
    auto stop_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop_time - start_time);
    // Close the filestream
    file.close();
    // Print required outputs
    cout << "\n~o~ For m=" << m << " and p=" << p << " ~o~" << endl <<
    "* Number of taxi additions: " << taxi_additions << endl <<
    "* Number of distance updates: " << distance_updates << endl <<
    "* Number of successful taxi calls: " << taxi_call << endl <<
    "* Number of operations skipped\ndue to no taxis being available: " << empty_heap << endl <<
    "* Elapsed time of execution: " << duration.count() << " microseconds" << endl;

    return 0;
}