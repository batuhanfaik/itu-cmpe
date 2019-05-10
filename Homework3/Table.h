/* @Author
 *
 * Student Name: Batuhan Faik Derinbay 
 * Student ID: 150180705
 * Date: 01/05/19
*/
#ifndef OOP_HW3_TABLE_H
#define OOP_HW3_TABLE_H

#include "Product.h"
#include <string>
#include <iostream>

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
    void print() const{
        cout << "Table number: " << table_number << endl;
        cout << "Number of orders: " << order_amount << endl;
        cout << "Total price of the table: " << total_price << endl;
        cout << "   Ordered products: " << endl;
        for (int i = 0; i < order_amount; ++i) {
            product_list[i].print();
        }
        cout << "   Ordered product amounts: " << endl;
        for (int j = 0; j < order_amount; ++j) {
            cout << product_amount_list[j] << endl;
        }
    }
    
    int get_order_amount() {
        return order_amount;
    }
    
    Product* get_product_list() {
        return product_list;
    }
    
    int* get_product_amount_list() {
        return product_amount_list;
    }
};


#endif //OOP_HW3_TABLE_H
