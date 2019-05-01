/* @Author
 *
 * Student Name: Batuhan Faik Derinbay 
 * Student ID: 150180705
 * Date: 01/05/19
*/
#include <iostream>
#include "Ingredient.h"

using namespace std;

void Ingredient::print() const{
    cout << "Ingredient name: " << name << endl;
}

void Type1::print() const{
    Ingredient::print();
    cout << "Amount available: " << item_weight << " grams" << endl;
    cout << "Price: " << price_per_gram << " TL" << endl;
}

void Type2::print() const{
    Ingredient::print();
    cout << "Amount available: " << number << " units" << endl;
    cout << "Price: " << price_per_unit << " TL" << endl;
}

void Type3::print() const{
    Ingredient::print();
    cout << "Amount available: " << milliliter << " ml" << endl;
    cout << "Price: " << price_per_milliliter << " TL" << endl;
}