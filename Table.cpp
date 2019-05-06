/* @Author
 *
 * Student Name: Batuhan Faik Derinbay 
 * Student ID: 150180705
 * Date: 01/05/19
*/
#include "Table.h"
#include "Product.h"
#include <string>
#include <iostream>

using namespace std;

void Table::print() const{
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