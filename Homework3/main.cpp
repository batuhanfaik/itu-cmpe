#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdio>
#include <cstring>

using namespace std;

class MobileHost {
    int id;
    int parent_bs_id;
public:
    MobileHost *next;

    MobileHost();

    MobileHost(int, int);

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

MobileHost::~MobileHost() {
    // TODO Implement destructor
}

MobileHost::MobileHost() {
    id = -1;
    parent_bs_id = -1;
    next = NULL;
}

MobileHost::MobileHost(int id_in, int bs_id_in) {
    this->id = id_in;
    this->parent_bs_id = bs_id_in;
    next = NULL;
}

class BaseStation {
    int id;
    int parent_id;
public:
    BaseStation *next;
    BaseStation *child;
    MobileHost *mh_child;

    BaseStation();

    BaseStation(int, int);

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

BaseStation::~BaseStation() {
    // TODO Implement destructor
}

BaseStation::BaseStation() {
    this->id = -1;
    this->parent_id = -1;
    next = NULL;
    child = NULL;
    mh_child = NULL;
}

BaseStation::BaseStation(int id_in, int parent_id_in) {
    this->id = id_in;
    this->parent_id = parent_id_in;
    next = NULL;
    child = NULL;
    mh_child = NULL;
}

class Stack {
    int top;
    int size;
public:
    BaseStation **stack_array;

    void initialize(int);

    bool push(BaseStation *);

    bool is_empty();

    BaseStation *pop();

    BaseStation *peek();

    ~Stack();
};

void Stack::initialize(int size_in) {
    this->size = size_in;
    this->stack_array = new BaseStation *[size_in];
    this->top = 0;
}

bool Stack::push(BaseStation *x) {
    if (top >= size - 1) {
        cout << "Stack Overflow!" << endl;
        return false;
    } else {
        stack_array[++top] = x;
        cout << x->get_id() << " is pushed into stack." << endl;
        return true;
    }
}

bool Stack::is_empty() {
    return (top < 0);
}

BaseStation *Stack::pop() {
    if (top < 0) {
        cout << "Stack Underflow!" << endl;
        return NULL;
    } else {
        BaseStation *x = stack_array[top--];
        return x;
    }
}

BaseStation *Stack::peek() {
    if (top < 0) {
        cout << "Stack is Empty!" << endl;
        return NULL;
    } else {
        BaseStation *x = stack_array[top];
        return x;
    }
}

Stack::~Stack() {
    delete[] stack_array;
}

class MobileNetwork {
    int bs_amount;
    int mh_amount;
    BaseStation *top;
public:
    MobileNetwork();

    static bool is_visited(const int *, int, int);

    BaseStation *find_bs(int);

    void add_bs(BaseStation &);

    void add_mh(MobileHost &);
};

MobileNetwork::MobileNetwork() {
    top = new BaseStation(0, 0);
    bs_amount = 1;
    mh_amount = 0;
}

bool MobileNetwork::is_visited(const int *visited_list, int visited_amount, int search_id) {
    int index = 0;
    while (index < visited_amount) {
        if (visited_list[index] == search_id) {
            return true;
        }
        index++;
    }
    return false;
}

BaseStation *MobileNetwork::find_bs(int bs_id) {
    BaseStation *current_bs = top;
    // Initialization steps for DFS
    Stack s = Stack();
    s.initialize(bs_amount);    // Size can be the sum of all the nodes at most
    int visited[bs_amount];
    int visited_amount = 1;
    visited[0] = 0;     // Central controller is the first node
    s.push(current_bs);
    // Start searching
    while (!s.is_empty()) {
        current_bs = s.pop();
        if (current_bs->get_id() == bs_id) {
            return current_bs;      // Base station found
        }
        if (!is_visited(visited, visited_amount, current_bs->get_id())) {
            visited[visited_amount] = current_bs->get_id();      // Mark the node as seen
            visited_amount++;
        }
        if (current_bs->child) {
            current_bs = current_bs->child;
            while (current_bs) {     // Look for children at the same level
                if (!is_visited(visited, visited_amount, current_bs->get_id())) {    // If not seen before push
                    s.push(current_bs);
                }
                current_bs = current_bs->next;      // Next station at the same level
            }
        }
    }
    return NULL;    // Couldn't find the base station (hope not)
}

void MobileNetwork::add_bs(BaseStation &bs) {
    BaseStation *parent_bs = find_bs(bs.get_parent_id());
    if (!parent_bs) {    // If can't find the parent
        cout << "Can't find your parent BS!" << endl;
    } else {
        if (!parent_bs->child) {     // Input base station is the new child if parent has no child
            parent_bs->child = &bs;
        } else {
            BaseStation *current_bs = parent_bs->child;
            while (current_bs->next) {       // Find the last child
                current_bs = current_bs->next;
            }
            current_bs->next = &bs;     // Input base station is added end of the level
        }
    }
}

void MobileNetwork::add_mh(MobileHost &mh) {
    BaseStation *parent_bs = find_bs(mh.get_parent_bs_id());
    if (!parent_bs) {    // If can't find the parent
        cout << "Can't find your parent BS!" << endl;
    } else {
        if (!parent_bs->mh_child) {     // Input mobile host is the new mh child if parent has no mh child
            parent_bs->mh_child = &mh;
        } else {
            MobileHost *current_mh = parent_bs->mh_child;
            while (current_mh->next) {       // Find the last mobile host
                current_mh = current_mh->next;
            }
            current_mh->next = &mh;     // Input mobile host is added end of mobile hosts
        }
    }
}

struct Message {
    string msg;
    int receiver;
};

int main(int argc, char** argv) {
    // TODO Input files as command line argument
    // TODO Read files
    // TODO Structure the network
    // TODO DFS the network
    // TODO Print paths
    // Get user inputs via CLI
//    string networks_file = argv[1];
//    string messages_file = argv[2];
    string networks_file = "Network.txt";
    string messages_file = "Messages.txt";
    // Open input file stream for networks file
    ifstream networks(networks_file);
    string type; int id; int parent_id;     // Declare types for reading the networks file
    if (networks.is_open()) {
        while (networks >> type >> id >> parent_id){
//        cout << "Type: " << type << "\tID: " << id << "\tParent ID: " << parent_id << endl;
        }
    } else {
        cout << "Unable to open " << networks_file << " file!" << endl;
    }
    networks.close();
    // Open input file stream for networks file
    ifstream messages(messages_file);
    string line; string part; string msg; int i; int receiver = 0;     // Declare types for reading the messages file
    if (messages.is_open()) {
        while (getline(messages, line)) {    // Get the line
            stringstream ss(line);
            i = 0;
            while (getline(ss, part, '>')){      // Parse it using the delimiter '>'
                if (part == "\r" || part == "\n"){      // Check if there are any abnormalities
                    i = 2;
                }
                if (i == 0){    // First item is the message
                    msg = part;
                    i++;
                } else if (i == 1){     // Second item is the receiver id
                    receiver = stoi(part);
                }
            }
            if (i < 2){     // Add only if there were no abnormalities like empty line or carriage return
                cout << "Message: " << msg << "\tReceiver: " << receiver << endl;
            }
        }
    } else {
        cout << "Unable to open " << messages_file << " file!" << endl;
    }
    messages.close();

    return 0;
}
