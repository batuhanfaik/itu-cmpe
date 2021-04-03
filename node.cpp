#include "node.h"

Node::Node(int dla) {
  this->distinct_letter_amount = dla;
  this->parent = nullptr;
  this->index = 0;
  this->is_leaf = false;
  this->children = vector<Node*>(10);
  this->data = vector<vector<bool>>(dla);

  // Falsify the data matrix
  for (int i = 0; i < dla; i++) {
    data[i] = vector<bool>(10, false);
  }
}

Node::Node(int dla, Node* parent, int row, int col) {
  this->distinct_letter_amount = dla;
  this->parent = parent;
  this->index = parent->index * 10 + col + 1;
  this->is_leaf = (row == dla - 1);
  this->children = vector<Node*>(10);
  this->data = parent->data;
  this->data[row][col] = true;
}

vector<Node *> Node::get_children() const {
  return children;
}

void Node::set_children(vector<Node*> c) {
  this->children = c;
}

vector<vector<bool>> Node::get_data() {
  return data;
}

void Node::print() {
  cout << "Node index: " << index << endl;
  cout << "Is leaf: " << is_leaf << endl;
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
  cout << endl;
}

int Node::get_index() const {
  return index;
}
bool Node::leaf() const {
  return is_leaf;
}
