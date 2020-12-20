//
// Created by batuhanfaik on 20/12/2020.
//

#ifndef HW2_HEAP_H
#define HW2_HEAP_H

#include <vector>
#include "taxi.h"

using namespace std;

class Heap {
    vector<Taxi> taxis;
    int size;

    static int get_parent(int);

    static int get_left(int);

    static int get_right(int);

    void compare_parents(int);

    void min_heapify(int);

public:
    Heap();

    void add_taxi(Taxi);

    void update_random_taxi(int, double);

    double call_taxi();

    int get_size() const;

    ~Heap();
};

#endif //HW2_HEAP_H
