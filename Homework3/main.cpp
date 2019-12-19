#include <iostream>

using namespace std;

class Stack{
    int top;
    int size;
public:
    int* stack_array;
    void initialize(int);
    bool push(int);
    bool is_empty();
    int pop();
    int peek();
    ~Stack();
};

void Stack::initialize(int size_in) {
    this->size = size_in;
    this->stack_array = new int[size_in];
    this->top = 0;
}

bool Stack::push(int x){
    if (top >= size-1){
        cout << "Stack Overflow!" << endl;
        return false;
    } else {
        stack_array[++top] = x;
        cout << x << " is pushed into stack." << endl;
        return true;
    }
}

bool Stack::is_empty() {
    return (top < 0);
}

int Stack::pop() {
    if (top < 0){
        cout << "Stack Underflow!" << endl;
        return 0;
    } else {
        int x = stack_array[top--];
        return x;
    }
}

int Stack::peek() {
    if (top < 0){
        cout << "Stack is Empty!" << endl;
        return 0;
    } else {
        int x = stack_array[top];
        return x;
    }
}

Stack::~Stack() {
    delete [] stack_array;
}

class Queue{
    // TODO
};

class MobileNetwork{
    // TODO
};

class BaseStation{
    // TODO
    int id;
    int parent_id;
public:
    ~BaseStation();
    int get_id();
    void set_id(int);
    int get_parent_id();
    void set_parent_id(int);
};

int BaseStation::get_id() {
    return id;
}

void BaseStation::set_id(int x) {
    id = x;
}

int BaseStation::get_parent_id() {
    return parent_id;
}

void BaseStation::set_parent_id(int x) {
    parent_id = x;
}

BaseStation::~BaseStation(){
    // TODO Implement destructor
}

class MobileHost{
    // TODO
    int id;
    int parent_bs_id;
public:
    ~MobileHost();
    int get_id();
    void set_id(int);
    int get_parent_bs_id();
    void set_parent_bs_id(int);
};

int MobileHost::get_id() {
    return id;
}

void MobileHost::set_id(int x) {
    id = x;
}

int MobileHost::get_parent_bs_id() {
    return parent_bs_id;
}

void MobileHost::set_parent_bs_id(int x) {
    parent_bs_id = x;
}

MobileHost::~MobileHost(){
    // TODO Implement destructor
}

int main() {
    // TODO Input files as command line argument
    // TODO Read files
    // TODO Structure the network
    // TODO DFS the network
    // TODO Print paths
    return 0;
}
