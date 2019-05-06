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