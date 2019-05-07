/* @Author
 *
 * Student Name: Batuhan Faik Derinbay 
 * Student ID: 150180705
 * Date: 01/05/19
*/
#ifndef OOP_HW3_PRODUCT_H
#define OOP_HW3_PRODUCT_H

#include <string>
#include "Ingredient.h"

using namespace std;

class Product {
    string name;
    int ingredient_count;
    Ingredient** ingredient_list;
    float total_price;
public:
    Product():name(""),ingredient_count(0),ingredient_list(nullptr),total_price(0){};
    Product(string name, float total_price):
        name(name),total_price(total_price),ingredient_count(0),ingredient_list(nullptr){};
    Product(string name, int ingredient_count, Ingredient** ingredient_list, float total_price):
        name(name),ingredient_count(ingredient_count),ingredient_list(ingredient_list),total_price(total_price){};
    void print();
    string get_name();
    float get_price();
    int get_ingredient_count();
    Ingredient** get_ingredient_list();
    ~Product();
};


#endif //OOP_HW3_PRODUCT_H
