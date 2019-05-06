/* @Author
 *
 * Student Name: Batuhan Faik Derinbay 
 * Student ID: 150180705
 * Date: 01/05/19
*/
#ifndef OOP_HW3_TABLE_H
#define OOP_HW3_TABLE_H

#include <string>
#include "Product.h"

using namespace std;

class Table {
    int table_number;
    int  order_amount;
    Product* product_list;
    int* product_amount_list;
    float total_price;
public:
    Table():table_number(0),order_amount(0),product_list(nullptr),product_amount_list(nullptr),total_price(0){} //Default constructor
    Table(int table_number, int order_amount, Product* product_list, int* product_amount_list, float total_price):
        table_number(table_number),order_amount(order_amount),product_list(product_list),
        product_amount_list(product_amount_list),total_price(total_price){};
    void print() const;
    int get_order_amount();
    Product* get_product_list();
    int* get_product_amount_list();
};


#endif //OOP_HW3_TABLE_H
