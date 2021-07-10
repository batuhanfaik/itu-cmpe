// This homework can be compiled using the following command
// g++ -o a.out main.cpp sale.cpp quicksort.cpp

#include<iostream>
#include<fstream>
#include<string>
#include <vector>
#include <chrono>   // Required to measure time

#include "sale.h"
#include "quicksort.h"

using namespace std;

bool MEASURE_TIME = true;
bool PRINT_SORTED = false;

int main(int argc, char** argv) {
    int N = 10;
    if (argc < 2){
        cout << "An N value needs to be passed in.\nThis run will assume that N=10" << endl;
    } else if (argc > 2){
        cout << "More than one parameters are passed in.\nThis run will assume that N=10" << endl;
    } else {
        N = stoi(argv[1]);
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

    // Measure time of the sorting algorithm or sort only
    if (MEASURE_TIME){
        auto start_time = chrono::high_resolution_clock::now();
        quicksort(sale_items, 0, N - 1);
        auto stop_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop_time - start_time);
        cout << "For N=" << N << endl << "Elapsed time of execution: " << duration.count() << " microseconds" << endl;
    } else {
        quicksort(sale_items, 0, N - 1);
    }

    if (PRINT_SORTED){
        // Print sale items using a range based for loop
        for (const auto & sale_item : sale_items) {
            sale_item.print();
        }
    }

    // Write to file
    ofstream sorted ("sorted.txt");
    if (sorted.is_open()){
        sorted << "Country\t"<<"Item Type\t"<<"Order ID\t"<<"Units Sold\t"<<"Total Profit\n";
        for (const auto & sale_item : sale_items) {
            sorted << sale_item;
        }
        sorted.close();
    }
    // Free up the memory
    sale_items.erase(sale_items.begin(), sale_items.end());

    return 0;
}
