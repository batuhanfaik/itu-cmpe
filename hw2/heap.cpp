//
// Created by batuhanfaik on 20/12/2020.
//

#include <algorithm>
#include "heap.h"
#include "taxi.h"

// PRIVATE METHODS
int Heap::get_parent(int index) {
    return int((index - 1)/2);
}

int Heap::get_left(int index) {
    return (index * 2 + 1);
}

int Heap::get_right(int index) {
    return (index * 2 + 2);
}

void Heap::compare_parents(int starting_index) {
    int index = starting_index;
    // Bubble upwards until min heap is satisfied
    while (index != 0 && taxis.at(get_parent(index)).get_distance() > taxis.at(index).get_distance()) {
        swap(taxis.at(index), taxis.at(get_parent(index)));
        index = get_parent(index);
    }
}

void Heap::min_heapify(int index) {
    int left_index = get_left(index);
    int right_index = get_right(index);
    int min_distance_index = index;

    // Check which child is smaller (has less distance)
    if (left_index < taxis.size() && taxis.at(left_index).get_distance() < taxis.at(index).get_distance()){
        min_distance_index = left_index;    // Left child is closer to the hotel
    }
    if (right_index < taxis.size() && taxis.at(right_index).get_distance() < taxis.at(min_distance_index).get_distance()){
        min_distance_index = right_index;    // Right child is closer to the hotel
    }
    if (min_distance_index != index){    // Heapify subtree
        swap(taxis.at(index), taxis.at(min_distance_index));
        min_heapify(min_distance_index);
    }
}

// PUBLIC METHODS
Heap::Heap(){
    size = 0;
}

void Heap::add_taxi(Taxi new_taxi) {
    // Add new taxi to the list
    taxis.push_back(new_taxi);
    size = taxis.size();
    // Get its index
    int index = int(taxis.size() - 1);
    compare_parents(index);
}

void Heap::update_random_taxi(int index, double decrease_distance) {
    // Decrease the distance of the taxi at the index by 0.01
    taxis.at(index).set_distance(taxis.at(index).get_distance() - decrease_distance);
    compare_parents(index);
}

double Heap::call_taxi() {
    // Store the distance of the taxi in a local variable
    double distance = taxis.front().get_distance();
    // Print the taxi that is being called
    taxis.front().print();
    // Delete the taxi from the list
    taxis.erase(taxis.begin());
    size = taxis.size();
    // Reorder the heap (heapify)
    min_heapify(0);
    return distance;
}

int Heap::get_size() const {
    return size;
}

Heap::~Heap() {
    taxis.erase(taxis.begin(), taxis.end());
}
