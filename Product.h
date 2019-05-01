/* @Author
 *
 * Student Name: Batuhan Faik Derinbay 
 * Student ID: 150180705
 * Date: 01/05/19
*/
#ifndef OOP_HW3_PRODUCT_H
#define OOP_HW3_PRODUCT_H

#include <string>

using namespace std;

class Product {
    string name;
    string* ingredient;
    int* ingredient_amount;
    float total_price;
};


#endif //OOP_HW3_PRODUCT_H
