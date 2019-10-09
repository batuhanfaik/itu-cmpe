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
    void sort();
    void sell(int);
    void current_stock();
    void clear();
};

void Stock::create() {
    this->head = nullptr;
}

void Stock::add_stock(int shoe_info) {
    // If this is the first item
    if (this->head == nullptr){
        head = new Node;
        head->size = shoe_info;
        head->quantity = 1;
        head->next = nullptr;
    } else {
        // Find a node with matching shoe_info
        Node* matching_node = head;
        Node* end_node = head->next;
        bool matching_node_found = false;
        while (!matching_node_found && matching_node != nullptr){
            if (matching_node->size == shoe_info){
                matching_node_found = true;
            } else {
                // Find the end node for later use
                if (matching_node->next == nullptr){
                    end_node = matching_node;
                }
                matching_node = matching_node->next;
            }
        }
        if (!matching_node_found){  // Create a new shoe node
            Node* new_entry = new Node;
            new_entry->size = shoe_info;
            new_entry->quantity = 1;
            new_entry->next = nullptr;
            end_node->next = new_entry;
        } else {    // There exists such shoe size, so update the quantity
            matching_node->quantity++;
        }
    }
}

void Stock::sell(int shoe_info) {
    // Find a node with matching shoe_info
    Node* matching_node = head;
    Node* prev_node = nullptr;
    bool matching_node_found = false;
    while (!matching_node_found && matching_node != nullptr){
        if (matching_node->size == shoe_info){
            matching_node_found = true;
        } else {
            // Find the end node for later use
            prev_node = matching_node;
            matching_node = matching_node->next;
        }
    }
    if (!matching_node_found){  // If no node is available, there aren't any left in the stock
        cout << "NO_STOCK" << endl;
    } else {    // There exists such shoe size, so update the quantity
        if (matching_node->quantity > 0){
            matching_node->quantity--;
        } else cout << "NO_STOCK" << endl;
    }
}

void Stock::current_stock() {
    // Go through all of the nodes and print
    Node* matching_node = head;
    while (matching_node != nullptr){
        cout << matching_node->size << ':' << matching_node->quantity << endl;
        matching_node = matching_node->next;
    }
}

void Stock::sort() {
    int min_size = 99999;
    Node* current_node = head;
    Node* min_node = head;
    Node* prev_node = nullptr;
    // Find the smallest shoe size given
    while (current_node != nullptr){
        if (current_node->size < min_size){
            min_size = current_node->size;
            min_node = current_node;
        }
        prev_node = current_node;
        current_node = current_node->next;
    }
    // Swap the head and min_node
    Node* tmp_node = head;
    Node* tmp_next = head->next;
    head = min_node;
    min_node = tmp_node;
    head->next = min_node->next;
    min_node->next = tmp_next;

    // Bubble sort
    current_node = head->next;
    Node* next_node;
    if (current_node != nullptr){
        next_node = current_node->next;
    } else {
        next_node = current_node;
    }
    while (next_node != nullptr){
        while (current_node->next != nullptr){
            if (current_node->size > next_node->size){
                tmp_node = current_node;
                tmp_next = next_node->next;
                current_node = next_node;
                next_node = tmp_node;
                current_node->next = next_node->next;
                next_node->next = tmp_next;
            }
            current_node = current_node->next;
        }
        next_node = current_node->next;
    }
}

void Stock::clear() {
    // Go through all the nodes and delete
    Node* matching_node = head;
    while (matching_node != nullptr){
        delete matching_node;
    }
}

int main() {
    Stock my_stock{};

    // Read the input file and append operations to a list
    string shoe_info;
    int *operation_list;
    int no_of_operations = 0;
    ifstream stock_file("input.txt");
    if (stock_file.is_open()) {
        while (getline(stock_file, shoe_info)) {
            stringstream ss(shoe_info);
            while (getline(ss, shoe_info, ' ')){
                no_of_operations++;
            }
        }
        operation_list = new int[no_of_operations];
        int index = 0;
        stock_file.clear();     // clear fail and eof bits
        stock_file.seekg(0, ios::beg);      // back to the start!
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

    my_stock.create();

    // Check if operation_list is successfully created
    if (operation_list != nullptr) {
        // Recurse through the operations
        for (int i = 0; i < no_of_operations; ++i) {
            if (operation_list[i] > 0) {
                my_stock.add_stock(operation_list[i]);
                my_stock.sort();
            } else if (operation_list[i] < 0){
                my_stock.sell(operation_list[i]);
            } else if (operation_list[i] == 0){
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