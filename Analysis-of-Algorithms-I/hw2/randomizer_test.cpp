#include <iostream>
#include <chrono>
#include <random>
#include <stdlib.h>

using namespace std;

template <typename T>
T get_random_int(T start, T end) {
    random_device rand_device;
    mt19937 generator(rand_device());
    uniform_int_distribution<T> distribution(start, end);
    return distribution(generator);
}

int main(){
    int N = 100000;
    srand(time(nullptr));

    auto start_time = chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        rand() % 100 + 1;
    }
    auto stop_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop_time - start_time);
    cout << "rand(): " << duration.count() << " microseconds" << endl;

    start_time = chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        get_random_int(0, 100);
    }
    stop_time = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(stop_time - start_time);
    cout << "mt: " << duration.count() << " microseconds" << endl;

    return 0;
}