/* @Author
 *
 * Student Name: Batuhan Faik Derinbay 
 * Student ID: 150180705
 * Date: 01/05/19
*/
#ifndef OOP_HW3_INGREDIENT_H
#define OOP_HW3_INGREDIENT_H

#include <string>

using namespace std;

class Ingredient {
    string name;
    int item_count;
    float price;
public:
    virtual void print();
};

class Type1: public Ingredient{
public:
    void print();
};

class Type2: public Ingredient{
public:
    void print();
};

class Type3: public Ingredient{
public:
    void print();
};

#endif //OOP_HW3_INGREDIENT_H
