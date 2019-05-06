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
public:
    Ingredient():name(""){};
    Ingredient(string& name):name(name){};
    string get_name() const;
    virtual void print() const;
    virtual float get_price() const;
    virtual void const set_price(float){};
};

class Type1: public Ingredient{
    int item_weight;
    float price_per_gram;
public:
    Type1():Ingredient(),item_weight(0),price_per_gram(0){};
    Type1(string& name, int item_weight):
        Ingredient(name),item_weight(item_weight),price_per_gram(0){};
    Type1(string& name, int item_weight, float price_per_gram):
        Ingredient(name),item_weight(item_weight),price_per_gram(price_per_gram){};
    void print() const;
    float get_price() const;
    void const set_price(float);
};

class Type2: public Ingredient{
    int number;
    float price_per_unit;
public:
    Type2():Ingredient(),number(0),price_per_unit(0){};
    Type2(string& name, int number):
            Ingredient(name),number(number),price_per_unit(0){};
    Type2(string& name, int number, float price_per_unit):
        Ingredient(name),number(number),price_per_unit(price_per_unit){};
    void print() const;
    float get_price() const;
    void const set_price(float);
};

class Type3: public Ingredient{
    int milliliter;
    float price_per_milliliter;
public:
    Type3():Ingredient(),milliliter(0),price_per_milliliter(0){};
    Type3(string& name, int milliliter):
        Ingredient(name),milliliter(milliliter),price_per_milliliter(0){};
    Type3(string& name, int milliliter, float price_per_milliliter):
        Ingredient(name),milliliter(milliliter),price_per_milliliter(price_per_milliliter){};
    void print() const;
    float get_price() const;
    void const set_price(float);
};

#endif //OOP_HW3_INGREDIENT_H
