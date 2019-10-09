#include <iostream>
#include <fstream>
#include <string>

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
        int node_index = 0;
        while (!matching_node_found && matching_node != nullptr){
            node_index++;
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
            end_node->next = new_entry;
        } else {    // There exists such shoe size, so update the quantity
            matching_node->quantity++;
        }
    }

    Node* new_entry = new Node;
    new_entry->size
}

int main() {
    Stock my_stock;

    // Read the input file and append operations to a list
    string shoe_info;
    int *operation_list;
    int no_of_operations = 0;
    ifstream stock_file("input.txt");
    if (stock_file.is_open()) {
        while (getline(stock_file, shoe_info, ' ')) {
            no_of_operations++;
        }
        operation_list = new int[no_of_operations];
        int index = 0;
        while (getline(stock_file, shoe_info, ' ')) {
            operation_list[index++] = stoi(shoe_info);
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
            } else if (operation_list[i] < 0){

            } else if (operation_list[i] == 0){

            } else {
                cout << "Unknown operation" << endl;
            }
        }
    }

    return 0;
}