/* @Author
 *
 * Student Name: Batuhan Faik Derinbay 
 * Student ID: 150180705
 * Date: 01/05/19
*/
#ifndef OOP_HW3_TABLE_H
#define OOP_HW3_TABLE_H

#include <string>

using namespace std;

class Table {
    int  people_amount;
    string* orders;
    int* orders_amount;
public:
    void print();
};


#endif //OOP_HW3_TABLE_H
