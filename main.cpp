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
#include <cstdio>
#include <bits/stdc++.h>
#include "Ingredient.h"
#include "Product.h"
#include "Table.h"

using namespace std;

void fix_file(string);
int const count_stock();
int const count_menu();
int const count_ingredients(string&);
void const read_stock(Ingredient**);
void const read_menu(Product*, Ingredient**);
void const read_tables(Table*);

//See the source information below for split functions
vector<string> split(string, char);
vector<string> split(string, string);


int main(){
    try{ //Fix input files (Replace "\r" with "")
        fix_file("stock.txt");
        fix_file("menu.txt");
        fix_file("order.txt");
    } catch(const char* err_msg){
        cout << err_msg << endl;
        return EXIT_FAILURE;
    }

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

    //Get the menu ready
    int product_amount;
    try{
        product_amount = count_menu();
    } catch(const char* err_msg){
        cout << err_msg << endl;
        return EXIT_FAILURE;
    }
    Product* product_list;
    product_list = new Product[product_amount]; //Allocate memory for the menu_list
    read_menu(product_list, stock_list);

    //Set the tables up
    Table tables[5];
    read_tables(tables);

    //Bloopers
    cout << "I AM ALIVE" << endl;
    return 0;
}

void find_replace(string& subject, const string& search,
                          const string& replace) {
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != string::npos) {
        subject.replace(pos, search.length(), replace);
        pos += replace.length();
    }
}

void fix_file(string file_name){
    string line;
    string sum;
    string str_temp;

    ifstream input_file; //Start input stream
    ofstream output_file("temp_fileout.txt"); //Temporary output file
    input_file.open(file_name); //Open the file

    if(!input_file.is_open() || !output_file) throw ("Error fixing the \"" + file_name + "\" file!");

    while (getline(input_file, line)){ //Get a line and replace "\r" with "" then output the line
        find_replace(line, "\r", "");
        str_temp = line;
        str_temp += "\n";
        output_file << str_temp;
    }

    int size = file_name.length();
    char file_name_array[size + 1]; //Declare a character array

    strcpy(file_name_array, file_name.c_str()); //Copy the contents of the string to character array

    rename("temp_fileout.txt", file_name_array); //Rename the output file to match the input file name
    input_file.close();
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

int const count_menu(){
    int product_amount = 0;
    string line;
    ifstream menu_file; //Start input stream
    menu_file.open("menu.txt"); //Open the file

    if (!menu_file.is_open()) throw "Error opening the \"menu.txt\" file!";

    while (getline(menu_file, line)) //If there is a line, increase the stock amount
        product_amount++;

    return product_amount-1; //Return 1 less because first line is ignored
}

int const count_ingredients(string& line){
    int ingredient_amount = 1;
    for(char& comma : line) { //Look for commas in the ingredients
        if(comma == ',') //If comma is found there is an ingredient
            ingredient_amount++;
    }
    return ingredient_amount;
}

//Title: How to Split a string using a char as delimiter
//Author: Varun
//Date: 26.01.2018
//Availability: https://thispointer.com/how-to-split-a-string-in-c
vector<string> split(string line, char delimiter){
    stringstream ss(line);
    string desired_data;
    vector<string> split_strings;

    while (getline(ss, desired_data, delimiter)){
        split_strings.push_back(desired_data);
    }

    return split_strings;
}

//Title: How to split a string by another string as delimiter
//Author: Varun
//Date: 26.01.2018
//Availability: https://thispointer.com/how-to-split-a-string-in-c
vector<string> split(string string_to_be_split, string delimeter){
    vector<string> split_string;
    int start_index = 0;
    int  end_index = 0;
    while( (end_index = string_to_be_split.find(delimeter, start_index)) < string_to_be_split.size() )
    {
        string val = string_to_be_split.substr(start_index, end_index - start_index);
        split_string.push_back(val);
        start_index = end_index + delimeter.size();
    }
    if(start_index < string_to_be_split.size())
    {
        string val = string_to_be_split.substr(start_index);
        split_string.push_back(val);
    }
    return split_string;
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
            cout << "Stock type of an item is not recognized!" << endl;
            stock_count--;
        }

        stock_count++; //Next item please :)
    }

//    //Print the stock
//    for (int i = 0; i < stock_count; ++i) {
//        stock_list[i]->print();
//    }
}

void const read_menu(Product* product_list, Ingredient** stock_list){
    int product_count = 0;
    string product_name;
    string ingredient_name;
    int ingredient_count; //Number of ingredients in a product
    int ingredient_amount; //Amount of the ingredient
    Ingredient** ingredient_list;

    //Parse the lines in stock.txt
    string line;
    ifstream menu_file("menu.txt"); //Open input stream
    getline(menu_file, line); //Ignore the first line

    while (getline(menu_file, line)) { //Read line by line
        float total_price = 0;
        vector<string> split_strings = split(line, '\t'); //Extract the name and ingredients
        product_name = split_strings[0];
        vector<string> all_ingredients = split(split_strings[1], ", "); //Split the ingredients

        ingredient_count = all_ingredients.size(); //Assign the number of ingredients
        ingredient_list = new Ingredient*[ingredient_count];

        for (int i = 0; i < ingredient_count; ++i) {
            vector<string> ingredient_split = split(all_ingredients[i], ' '); //Split the ingredient further
            if(ingredient_split[0] == "N/A"){ //If the menu item is not made up of ingredients
                ingredient_list[i] = new Ingredient(ingredient_name);
                ingredient_amount = 1;
            } else {
                ingredient_amount = stoi(ingredient_split[0]); //First value is the amount

                string type1_str = " gram ";
                string type3_str = " ml ";
                if (all_ingredients[i].find(type1_str) != string::npos){ //If "gram" is in the ingredient desc.
                    ingredient_name = ingredient_split[2]; //Append the name of the ingredient
                    for (int j = 3; j < ingredient_split.size(); ++j) { //For more than one word names, append the rest
                        ingredient_name.append(" ");
                        ingredient_name.append(ingredient_split[j]);
                    }
                    ingredient_list[i] = new Type1(ingredient_name, ingredient_amount); //Append to the list
                } else if(all_ingredients[i].find(type3_str) != string::npos){ //If "ml" is in the ingredient desc.
                    ingredient_name = ingredient_split[2]; //Append the name of the ingredient
                    for (int j = 3; j < ingredient_split.size(); ++j) { //For more than one word names, append the rest
                        ingredient_name.append(" ");
                        ingredient_name.append(ingredient_split[j]);
                    }
                    ingredient_list[i] = new Type3(ingredient_name, ingredient_amount); //Append to the list
                } else{
                    ingredient_name = ingredient_split[1]; //Append the name of the ingredient
                    for (int j = 2; j < ingredient_split.size(); ++j) { //For more than one word names, append the rest
                        ingredient_name.append(" ");
                        ingredient_name.append(ingredient_split[j]);
                    }
                    ingredient_list[i] = new Type2(ingredient_name, ingredient_amount); //Append to the list
                }
            }
            int stock_index = 0;
            int stock_count = count_stock();
            //Look for the ingredient with the matching name in the stock
            while (stock_index < stock_count && stock_list[stock_index]->get_name() != ingredient_name){
                stock_index++;
            }
            if(stock_index < stock_count){ //If in found in the stock
                total_price += stock_list[stock_index]->get_price()*ingredient_amount;
            } else{
                cout << "Ingredient can't be found in the stock." << endl;
            }
        }
        product_list[product_count] = Product(product_name, ingredient_count, ingredient_list, total_price);
        product_count++; //Increase the number of products
    }

    //Print the product list
    for (int k = 0; k < product_count; ++k) {
        product_list[k].print();
    }
}

void const read_tables(Table* tables){

}
