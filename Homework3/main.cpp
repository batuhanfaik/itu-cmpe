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
};

class MobileHost{
    // TODO
};

int main() {
    // TODO Input files as command line argument
    // TODO Read files
    // TODO Structure the network
    // TODO DFS the network
    // TODO Print paths
    return 0;
}
