/* @Author
Student Name: Batuhan Faik Derinbay
Student ID: 150180705
Date: 10.10.2019 */

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

struct Node {
    int size;
    int quantity;
    Node *next;
};

struct Stock {
    Node *head;

    void create();

    void add_stock(int);

    void sell(int);

    void current_stock();

    void clear();
};

void Stock::create() {
    this->head = nullptr;
}

void Stock::add_stock(int shoe_info) {
    // If this is the first item
    if (this->head == nullptr) {
        head = new Node;
        head->size = shoe_info;
        head->quantity = 1;
        head->next = nullptr;
    } else {
        // Find a node with matching shoe_info
        Node *matching_node = head;
        Node *end_node = head->next;
        Node *prev_node = nullptr;
        bool matching_node_found = false;
        while (!matching_node_found && matching_node != nullptr) {
            if (matching_node->size == shoe_info) {
                matching_node_found = true;
            } else {
                // Find the end node for later use
                if (matching_node->next == nullptr) {
                    end_node = matching_node;
                }
                matching_node = matching_node->next;
            }
        }
        if (!matching_node_found) {  // Create a new shoe node
            // Look for the smallest shoe size that is smaller than current and insert new node before
            matching_node = head;
            while (matching_node != nullptr && matching_node->size < shoe_info) {
                prev_node = matching_node;
                matching_node = matching_node->next;
            }
            if (matching_node != nullptr) {
                Node *new_entry = new Node;
                new_entry->size = shoe_info;
                new_entry->quantity = 1;
                new_entry->next = matching_node;
                if (prev_node == nullptr) {      // If there is no prev node, this is the head node
                    head = new_entry;
                } else {
                    prev_node->next = new_entry;
                }
            } else {    // Insert to the end because it's the largest shoe size
                Node *new_entry = new Node;
                new_entry->size = shoe_info;
                new_entry->quantity = 1;
                new_entry->next = nullptr;
                end_node->next = new_entry;
            }
        } else {    // There exists such shoe size, so update the quantity
            matching_node->quantity++;
        }
    }
}

void Stock::sell(int shoe_info) {
    // Find a node with matching shoe_info
    Node *matching_node = head;
    Node *prev_node = nullptr;
    bool matching_node_found = false;
    while (!matching_node_found && matching_node != nullptr) {
        if (matching_node->size == shoe_info) {
            matching_node_found = true;
        } else {
            // Find the end node for later use
            prev_node = matching_node;
            matching_node = matching_node->next;
        }
    }
    ofstream outfile;
    outfile.open("output.txt", ios_base::app);
    if (!matching_node_found) {  // If no node is available, there aren't any left in the stock
        cout << "NO_STOCK" << endl;
        outfile << "NO_STOCK" << endl;
    } else {    // There exists such shoe size, so update the quantity
        if (matching_node->quantity > 0) {
            matching_node->quantity--;
        } else if (matching_node->quantity == 0) {
            if (prev_node != nullptr){
                prev_node->next = matching_node->next;
            } else head = matching_node->next;
            delete matching_node;
            cout << "NO_STOCK" << endl;
            outfile << "NO_STOCK" << endl;
        } else {
            cout << "NO_STOCK" << endl;
            outfile << "NO_STOCK" << endl;
        }
    }
    outfile.close();
}

void Stock::current_stock() {
    // Go through all of the nodes and print
    Node *matching_node = head;
    ofstream outfile;
    outfile.open("output.txt", ios_base::app);
    while (matching_node != nullptr) {
        if (matching_node->quantity != 0) {
            cout << matching_node->size << ':' << matching_node->quantity << endl;
            outfile << matching_node->size << ':' << matching_node->quantity << endl;
        }
        matching_node = matching_node->next;
    }
    outfile.close();
}

void Stock::clear() {
    // Go through all the nodes and delete
    Node *matching_node = head;
    Node *tmp_next = head;
    while (tmp_next != nullptr) {
        matching_node = tmp_next;
        tmp_next = tmp_next->next;
        delete matching_node;
    }
}

int main(int argc, char** argv) {
    Stock my_stock{};
    string input_file_name = argv[1];
//    string input_file_name = "input2.txt";
    // Read the input file and append operations to a list
    string shoe_info;
    int *operation_list;
    int no_of_operations = 0;
    ifstream stock_file(input_file_name);
    if (stock_file.is_open()) {
        while (getline(stock_file, shoe_info)) {
            stringstream ss(shoe_info);
            while (getline(ss, shoe_info, ' ')) {
                no_of_operations++;
            }
        }
        operation_list = new int[no_of_operations];
        int index = 0;
        stock_file.clear();     // clear fail and eof bits
        stock_file.seekg(0, ios::beg);      // back to the start
        while (getline(stock_file, shoe_info)) {
            stringstream ss(shoe_info);
            while (getline(ss, shoe_info, ' ')) {
                operation_list[index++] = stoi(shoe_info);
            }
        }
        stock_file.close();
    } else {
        cout << "Unable to open file";
        operation_list = nullptr;
    }
    // Delete the existing output.txt file so the program won't append at the end of it
    remove("./output.txt");

    my_stock.create();

    // Check if operation_list is successfully created
    if (operation_list != nullptr) {
        // Recurse through the operations
        for (int i = 0; i < no_of_operations; ++i) {
            if (operation_list[i] > 0) {
                my_stock.add_stock(operation_list[i]);
            } else if (operation_list[i] < 0) {
                my_stock.sell(-operation_list[i]);
            } else if (operation_list[i] == 0) {
                my_stock.current_stock();
            } else {
                cout << "Unknown operation" << endl;
            }
        }
    }

    my_stock.clear();
    delete[] operation_list;
    return 0;
}