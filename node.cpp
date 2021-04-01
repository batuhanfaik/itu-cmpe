#include "node.h"

Node::Node(int dla) {
  this->distinct_letter_amount = dla;
  this->parent = nullptr;
  this->children = vector<Node*>(10);
  this->data = vector<vector<int>>(dla);

  // Zero out the data matrix
  for (int i = 0; i < dla; i++) {
    data[i] = vector<int>(10, 0);
  }
}

Node::Node(int dla, Node* parent, int row, int col) {
  this->distinct_letter_amount = dla;
  this->parent = parent;
  this->children = vector<Node*>(10);
  this->data = parent->data;
  this->data[row][col] = 1;
}

vector<Node *> Node::get_children() const {
  return children;
}

void Node::set_children(vector<Node*> c) {
  this->children = c;
}

vector<vector<int>> Node::get_data() {
  return data;
}

void Node::print() {
  for (int i = 0; i < 10; ++i) {
    cout << i << "\t";
  }
  cout << endl;
  for (int i = 0; i < distinct_letter_amount; i++) {
    for (int j = 0; j < 10; j++) {
      if (data[i][j]) {
        cout << data[i][j] << "\t";
      } else {
        cout << ".\t";
      }
    }
    cout << endl;
  }
  cout << endl << endl;
}