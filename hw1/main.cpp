#include<iostream>
#include<fstream>
#include<string>
#include <vector>   // Unnecessary flex

#include "sale.h"
#include "quicksort.h"

using namespace std;

int main(int argc, char** argv) {
    int N = 100;
    if (argc < 2){
        cout << "An N value needs to be passed in.\nThis run will assume that N=100" << endl;
    } else if (argc > 2){
        cout << "More than one parameters are passed in.\nThis run will assume that N=100" << endl;
    } else {
        N = stoi(argv[1]);
        cout << "N=" << N << endl;
    }

    ifstream file;
    file.open("sales.txt");

    if (!file) {
        cerr << "File cannot be opened!";
        exit(1);
    }

    string line, country, item_type, order_id;
	int units_sold;
	float total_profit;
	vector<Sale> sale_items;

    getline(file, line); //this is the header line

    // Read first N lines
    for (int i = 0; i < N; i++) {
        // Read values of the Sale object
        getline(file, country, '\t'); //country (string)
        getline(file, item_type, '\t'); //item type (string)
        getline(file, order_id, '\t'); //order id (string)
        file >> units_sold; //units sold (integer)
        file >> total_profit; //total profit (float)
        getline(file, line, '\n'); //this is for reading the \n character into dummy variable.

        Sale sale_item = Sale(country, item_type, order_id, units_sold, total_profit);
        sale_items.push_back(sale_item);
    }

    quickSort(sale_items, 0, N-1);

    // Print sale items using a range based for loop
    for (const auto & sale_item : sale_items) {
        cout << sale_item;
    }

    return 0;
}
