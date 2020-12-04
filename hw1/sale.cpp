//
// Created by batuhanfaik on 04/12/2020.
//

#include <iostream>
#include <utility>

#include "sale.h"

using namespace std;

Sale::Sale(string country, string item_type, string order_id, int units_sold, float total_profit) {
    this->country = move(country);
    this->item_type = move(item_type);
    this->order_id = move(order_id);
    this->units_sold = units_sold;
    this->total_profit = total_profit;
}

string Sale::get_country() {
    return this->country;
}

string Sale::get_item_type() {
    return this->item_type;
}

string Sale::get_order_id() {
    return this->order_id;
}

int Sale::get_units_sold() const {
    return this->units_sold;
}

float Sale::get_total_profit() const {
    return this->total_profit;
}

void Sale::print() {
    cout << "Sale Item\nCountry: " << this->country << endl << "Item Type: " << this->item_type << endl
    << "Order ID: " << this->order_id << endl << "Units Sold: " << this->units_sold << endl << "Total Profit: "
    << this->total_profit << endl;
}

ostream &operator<<(ostream & os, const Sale & sale) {  // Unnecessary flex
    return os << "Sale Item\nCountry: " << sale.country << endl << "Item Type: " << sale.item_type << endl
              << "Order ID: " << sale.order_id << endl << "Units Sold: " << sale.units_sold << endl << "Total Profit: "
              << sale.total_profit << endl;;
}
