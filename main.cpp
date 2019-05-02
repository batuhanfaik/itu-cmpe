/* @Author
 *
 * Student Name: Batuhan Faik Derinbay 
 * Student ID: 150180705
 * Date: 01/05/19
*/

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include "Ingredient.h"
#include "Product.h"
#include "Table.h"

using namespace std;

int const count_stock();
void const read_stock(Ingredient**);
void const read_tables(Table*);
vector<string> split(string, char);

int main(){
    //Fill the stock
    int stock_ingredient_amount;
    try{
        stock_ingredient_amount = count_stock();
    } catch(const char* err_msg){
        cout << err_msg << endl;
        return EXIT_FAILURE;
    }
    Ingredient** stock_list;
    stock_list = new Ingredient*[stock_ingredient_amount]; //Allocate memory for the stock_list
    read_stock(stock_list);

    //Set the tables up
    Table tables[5];
    read_tables(tables);

    //Bloopers
    cout << "I AM ALIVE" << endl;
    return 0;
}

int const count_stock(){
    int stock_amount = 0;
    string line;
    ifstream stock_file; //Start input stream
    stock_file.open("stock.txt"); //Open the file

    if (!stock_file.is_open()) throw "Error opening the \"stock.txt\" file!";

    while (getline(stock_file, line)) //If there is a line, increase the stock amount
        stock_amount++;

    return stock_amount-1; //Return 1 less because first line is ignored
}

vector<string> split(string line, char delimiter){
    stringstream ss(line);
    string desired_data;
    vector<string> split_strings;

    while (getline(ss, desired_data, delimiter)){
        split_strings.push_back(desired_data);
    }

    return split_strings;
}

void const read_stock(Ingredient** stock_list){
    int stock_count = 0;
    string ingredient_name;
    int ingredient_count;
    float ingredient_price;
    int ingredient_type;

    //Parse the lines in stock.txt
    string line;
    ifstream stock_file("stock.txt"); //Open input stream
    getline(stock_file, line); //Ignore the first line

    while (getline(stock_file, line)) { //Read line by line
        vector<string> split_strings = split(line, '\t'); //Values to variables
        ingredient_name = split_strings[0];
        ingredient_type = stoi(split_strings[1]);
        ingredient_count = stoi(split_strings[2]);
        ingredient_price = stof(split_strings[3]);

        if(ingredient_type == 1){ //Type1
            stock_list[stock_count] = new Type1(ingredient_name, ingredient_count, ingredient_price); //Append to the list
        } else if(ingredient_type == 2){ //Type2
            stock_list[stock_count] = new Type2(ingredient_name, ingredient_count, ingredient_price); //Append to the list
        } else if(ingredient_type == 3){ //Type3
            stock_list[stock_count] = new Type3(ingredient_name, ingredient_count, ingredient_price); //Append to the list
        } else{ //Type doesn't exist
            cout << "Stock type is not recognized!" << endl;
            stock_count--;
        }

        stock_count++; //Next item please :)
    }

    //Print the stock
    for (int i = 0; i < stock_count; ++i) {
        stock_list[i]->print();
    }
}

void const read_tables(Table* tables){

}
