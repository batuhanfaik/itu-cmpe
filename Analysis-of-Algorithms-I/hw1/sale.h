//
// Created by batuhanfaik on 04/12/2020.
//

#ifndef HW1_SALE_H
#define HW1_SALE_H

#include <string>

using namespace std;

class Sale {
    string country;
    string item_type;
    string order_id;
    int units_sold;
    float total_profit;
public:
    Sale(string, string, string, int, float);
    string get_country();
    string get_item_type();
    string get_order_id();
    int get_units_sold() const;
    float get_total_profit() const;
    void print() const;
    friend ostream &operator<<(ostream &, Sale const &);
};

#endif //HW1_SALE_H
