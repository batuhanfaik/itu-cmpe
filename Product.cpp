/* @Author
 *
 * Student Name: Batuhan Faik Derinbay 
 * Student ID: 150180705
 * Date: 01/05/19
*/
#include "Product.h"
#include <string>
#include <iostream>

using namespace std;

void Product::print() {
    cout << "Product name: " << name << endl;
    cout << "Number of ingredients: " << ingredient_count << endl;
    for (int i = 0; i < ingredient_count; ++i) {
        cout << "   Ingredient " << i+1 << ": " << endl;
        ingredient_list[i]->print();
    }
    cout << "Price of the product: " << total_price << endl;
}

string Product::get_name() {
    return name;
}

float Product::get_price() {
    return total_price;
}

int Product::get_ingredient_count() {
    return ingredient_count;
}

Ingredient** Product::get_ingredient_list() {
    return ingredient_list;
}

Product::~Product() {
//    for (int i = 0; i < ingredient_count; ++i) {
//        delete[] ingredient_list[i];
//    }
//    delete[] ingredient_list;
}