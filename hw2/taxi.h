//
// Created by batuhanfaik on 20/12/2020.
//

#ifndef HW2_TAXI_H
#define HW2_TAXI_H

#include <cmath>

using namespace std;

class Taxi {
    double longitude;
    double latitude;
    double distance_to_hotel;
public:
    Taxi(double, double, double, double);
    double calculate_distance_to_hotel(double, double) const;
    double get_distance() const;
    void print() const;
};


#endif //HW2_TAXI_H
