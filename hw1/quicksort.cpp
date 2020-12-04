/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * /
* @ Filename: quicksort
* @ Date: 04-Dec-2020
* @ AUTHOR: batuhanfaik
* @ Copyright (C) 2020 Batuhan Faik Derinbay
* @ Project: hw1
* @ Description: Not available
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <algorithm>

#include "quicksort.h"

using namespace std;

int partition(vector<Sale>& sale_items, int start_index, int end_index){
    Sale pivot_element = sale_items.at(end_index);
    int pivot_index = start_index;

    // For each element in the partition check if the item needs to be on the left or right
    for (int i = start_index; i < end_index; i++) {
        if (sale_items.at(i).get_units_sold() <= pivot_element.get_units_sold()) {
            swap(sale_items[i], sale_items[pivot_index]);
            pivot_index++;
        }
    }
    swap(sale_items.at(pivot_index), sale_items.at(end_index));

    return pivot_index;
}

void quickSort(vector<Sale>& sale_items, int start_index, int end_index){
    if (start_index < end_index){
        int partition_index = partition(sale_items, start_index, end_index);
        quickSort(sale_items, start_index, partition_index - 1);
        quickSort(sale_items, partition_index + 1, end_index);
    }
}