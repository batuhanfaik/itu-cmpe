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

string Ingredient::get_name() const {
    return name;
}

float Ingredient::get_price() const {
    return 0;
}

float Type1::get_price() const {
    return price_per_gram;
}

float Type2::get_price() const {
    return price_per_unit;
}

float Type3::get_price() const {
    return price_per_milliliter;
}

void const Type1::set_price(float x){
    price_per_gram = x;
}

void const Type2::set_price(float x){
    price_per_unit = x;
}

void const Type3::set_price(float x){
    price_per_milliliter = x;
}