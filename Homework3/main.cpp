/* @Author
Student Name: Batuhan Faik Derinbay
Student ID: 150180705
Date: 19.12.2019 */

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstring>

using namespace std;

class MobileHost {
    int id;
    int parent_bs_id;
public:
    MobileHost *next;

    MobileHost();

    MobileHost(int, int);

    int get_id();

    int get_parent_bs_id();
};

int MobileHost::get_id() {
    return id;
}

int MobileHost::get_parent_bs_id() {
    return parent_bs_id;
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
    BaseStation *parent;
    MobileHost *mh_child;

    BaseStation();

    BaseStation(int, int);

    int get_id();

    int get_parent_id();
};

int BaseStation::get_id() {
    return id;
}

int BaseStation::get_parent_id() {
    return parent_id;
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

    ~Stack();
};

void Stack::initialize(int size_in) {
    this->size = size_in;
    this->stack_array = new BaseStation *[size_in];
    this->top = 0;
}

bool Stack::push(BaseStation *x) {
    if (top >= size) {
        cout << "Stack Overflow!" << endl;
        return false;
    } else {
        stack_array[++top] = x;
//        cout << x->get_id() << " is pushed into stack." << endl;
        return true;
    }
}

bool Stack::is_empty() {
    return (top <= 0);
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

Stack::~Stack() {
    delete[] stack_array;
}

class Message {
    string msg;
    int receiver;
public:
    Message(string &, int);

    string get_msg();

    int get_receiver_id();
};

Message::Message(string &msg_in, int receiver_in) {
    this->msg = msg_in;
    this->receiver = receiver_in;
}

string Message::get_msg() {
    return this->msg;
}

int Message::get_receiver_id() {
    return this->receiver;
}

class MobileNetwork {
    int bs_amount;
    int mh_amount;
    BaseStation *receiver_bs;
    BaseStation *top;
public:
    MobileNetwork();

    static bool is_visited(const int *, int, int);

    BaseStation *find_bs(int);

    void add_bs(BaseStation &);

    void add_mh(MobileHost &);

    void print_bs(BaseStation &);

    void print_all(BaseStation &);

    string find_shortest_path(BaseStation &);

    void find_receiver(BaseStation &, int);

    void send_msg(Message &);

    void shutdown();
};

MobileNetwork::MobileNetwork() {
    top = new BaseStation(0, 0);
    bs_amount = 1;
    mh_amount = 0;
    receiver_bs = NULL;
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
            bs.parent = parent_bs;
        } else {
            BaseStation *current_bs = parent_bs->child;
            while (current_bs->next) {       // Find the last child
                current_bs = current_bs->next;
            }
            current_bs->next = &bs;     // Input base station is added end of the level
            bs.parent = parent_bs;
        }
        bs_amount++;
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
        mh_amount++;
    }
}

// Recursive DFS for printing base stations
void MobileNetwork::print_bs(BaseStation &bs) {
    if (bs.get_id() == -1) {
        print_bs(*top);
    } else {
        cout << "Now at BS: " << bs.get_id() << endl;
    }
    if (bs.child) {
        BaseStation *child_bs = bs.child;
        print_bs(*child_bs);
    }
    if (bs.next) {
        BaseStation *next_bs = bs.next;
        print_bs(*next_bs);
    }
}

// Recursive DFS for printing all nodes of the network
void MobileNetwork::print_all(BaseStation &bs) {
    if (bs.get_id() == -1) {
        print_all(*top);
    } else {
        cout << "Now at BS: " << bs.get_id() << endl;
    }
    if (bs.child) {
        BaseStation *child_bs = bs.child;
        print_all(*child_bs);
    }
    if (bs.next) {
        BaseStation *next_bs = bs.next;
        print_all(*next_bs);
    }
    if (bs.mh_child) {
        MobileHost *current_mh = bs.mh_child;
        while (current_mh) {
            cout << "Now at MH: " << current_mh->get_id() << endl;
            current_mh = current_mh->next;
        }
    }
}

// Find the shortest path for printing
string MobileNetwork::find_shortest_path(BaseStation &bs) {
    BaseStation *current_bs = &bs;
    string path;
    while (current_bs != top) {
        string id =  to_string(current_bs->get_id()) + " ";
        path.insert(0, id);
        current_bs = current_bs->parent;
    }
    path.insert(0, "0 ");
    return path;
}

// Recursive DFS search to find the receiver of the message
void MobileNetwork::find_receiver(BaseStation &bs, int receiver_id) {
    if (!receiver_bs) {
        cout << bs.get_id();
        if (bs.mh_child) {
            MobileHost *current_mh = bs.mh_child;
            while (current_mh) {
//            cout << "Now at MH: " << current_mh->get_id() << endl;
                if (current_mh->get_id() == receiver_id) {
                    cout << endl;
                    receiver_bs = &bs;
                }
                current_mh = current_mh->next;
            }
        }
        if (bs.child && !receiver_bs) {     // Go to children nodes
            cout << " ";
            BaseStation *child_bs = bs.child;
            find_receiver(*child_bs, receiver_id);
        }
        if (bs.next && !receiver_bs) {      // Go to sibling nodes
            cout << " ";
            BaseStation *next_bs = bs.next;
            find_receiver(*next_bs, receiver_id);
        }
    }
}

// Sending message
void MobileNetwork::send_msg(Message &message) {
    cout << "Traversing:";
    find_receiver(*top, message.get_receiver_id());      // Find the receiver
    if (!receiver_bs) {
        cout << endl << "Can not be reached the mobile host mh_" << message.get_receiver_id() << " at the moment"
             << endl;
    } else {
        cout << "Message:" << message.get_msg() << " To:" << find_shortest_path(*receiver_bs) << "mh_"
             << message.get_receiver_id() << endl;
    }
    receiver_bs = NULL;
}

void MobileNetwork::shutdown() {
    BaseStation *current_bs = top;
    BaseStation *to_delete = NULL;
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
        // Delete nodes
        to_delete = current_bs;
        if (to_delete) {
            // Go through all mobile hosts and delete them
            while (to_delete->mh_child) {
                MobileHost *mh_to_delete = to_delete->mh_child;
                to_delete->mh_child = mh_to_delete->next;
                delete mh_to_delete;
            }
            // Delete the base station
            delete to_delete;
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
}

int main(int argc, char **argv) {
    // Create the network
    MobileNetwork network = MobileNetwork();

    // Get user inputs via CLI
//    string networks_file = argv[1];
//    string messages_file = argv[2];
    string networks_file = "Network.txt";
    string messages_file = "Messages.txt";

    // Open input file stream for networks file
    ifstream networks(networks_file);
    string type;
    int id;
    int parent_id;     // Declare types for reading the networks file
    if (networks.is_open()) {
        while (networks >> type >> id >> parent_id) {
//        cout << "Type: " << type << "\tID: " << id << "\tParent ID: " << parent_id << endl;
            if (type == "BS") {      // Add base station
                auto *bs = new BaseStation(id, parent_id);
                network.add_bs(*bs);
            } else if (type == "MH") {   // Add mobile host
                auto *mh = new MobileHost(id, parent_id);
                network.add_mh(*mh);
            } else { cout << "Node type unknown!" << endl; }
        }
    } else {
        cout << "Unable to open " << networks_file << " file!" << endl;
    }
    networks.close();

    // Open input file stream for networks file
    ifstream messages(messages_file);
    string line;
    string part;
    string msg;
    int i;
    int receiver = 0;     // Declare types for reading the messages file
    if (messages.is_open()) {
        while (getline(messages, line)) {    // Get the line
            stringstream ss(line);
            i = 0;
            while (getline(ss, part, '>')) {      // Parse it using the delimiter '>'
                if (part == "\r" || part == "\n") {      // Check if there are any abnormalities
                    i = 2;
                }
                if (i == 0) {    // First item is the message
                    msg = part;
                    i++;
                } else if (i == 1) {     // Second item is the receiver id
                    receiver = stoi(part);
                }
            }
            if (i < 2) {     // Add only if there were no abnormalities like empty line or carriage return
//                cout << "Message: " << msg << "\tReceiver: " << receiver << endl;
                auto *message = new Message(msg, receiver);
                network.send_msg(*message);     // Send the message
                delete message;
            }
        }
    } else {
        cout << "Unable to open " << messages_file << " file!" << endl;
    }
    messages.close();

    // Garbage collection for nodes
    network.shutdown();

    return 0;
}
