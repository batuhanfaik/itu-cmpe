//
// Created by batuhanfaik on 20/12/2020.
//

#include <iostream>
#include "taxi.h"

using namespace std;

Taxi::Taxi(double longitude, double latitude, double hotel_long, double hotel_lat) {
    this->longitude = longitude;
    this->latitude = latitude;
    distance_to_hotel = calculate_distance_to_hotel(hotel_long, hotel_lat);
}

double Taxi::calculate_distance_to_hotel(double hotel_long, double hotel_lat) const {
    return sqrt(pow(longitude - hotel_long, 2) + pow(latitude - hotel_lat, 2));;
}

double Taxi::get_distance() const {
    return distance_to_hotel;
}

void Taxi::set_distance(double distance) {
    distance_to_hotel = distance;
}

void Taxi::print_called_taxi() const {
    cout << "A taxi is being called!\nDistance to hotel: " << distance_to_hotel << endl;
}
