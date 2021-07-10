/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * /
* @ Filename: main
* @ Date: 31-Mar-2021
* @ AUTHOR: batuhanfaik
* @ Copyright (C) 2021 Batuhan Faik Derinbay
* @ Project: hw1
* @ Description: Cryptarithmetic, to run: ./hw1 DFS TWO TWO FOUR outputFileName
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <iostream>
#include <string>
#include <chrono>   // Required to measure time
#include <vector>
#include <bits/stdc++.h>
#include <cmath>

using namespace std;


class Node {
  int distinct_letter_amount;
  int index;
  bool is_leaf;
  vector<vector<uint8_t>> data;
  Node *parent;
  vector<Node*> children;
 public:
  Node(int);

  Node(int, Node*, int, int);

  vector<Node*> get_children() const;

  void set_children(vector<Node*>);

  int get_index() const;

  bool leaf() const;

  vector<vector<uint8_t>> get_data();

  void print();
};


class Tree {
  int node_amount;
  int nodes_in_memory;
  Node* root;
  string letters;
  string operand1, operand2, result;
  vector<int> operand1_map, operand2_map, result_map;
  string solution, solution_to_file;

  string get_distinct_letters();
  vector<int> map_operand(string);
  void populate(int, Node*);
  bool satisfies_constraints(Node&);
  bool check_solution(Node&);
 public:
  Tree(const string&, const string&, const string&);

  int* bfs();

  int* dfs();

  int get_nodes_in_memory() const;

  string get_solution() const;

  string get_solution_to_file() const;
};


int main(int argc, char **argv) {
  // Get command line arguments
  string algorithm, operand1, operand2, result, output_file_name;
  if (argc == 6) {
    algorithm = argv[1];
    operand1 = argv[2];
    operand2 = argv[3];
    result = argv[4];
    output_file_name = argv[5];
  } else {
    cout << "5 arguments were expected, but got " << argc << ".\nExiting!" << endl;
    exit(1);
  }

  int* algo_results;
  // Create the tree
  Tree cryptarithmetic_tree = Tree(operand1, operand2, result);

  // Get the current time
  auto start_time = chrono::high_resolution_clock::now();

  // Run the search
  if (algorithm == "DFS")
    algo_results = cryptarithmetic_tree.dfs();
  else
    algo_results = cryptarithmetic_tree.bfs();

  // Searching ends
  auto stop_time = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::microseconds>(stop_time - start_time);

  // Write to file
  ofstream file;
  file.open(output_file_name + ".txt");
  file << cryptarithmetic_tree.get_solution_to_file();
  file.close();

  cout << "Algorithm: " << algorithm << endl <<
          "Number of visited nodes: " << algo_results[0] << endl <<
          "Maximum number of nodes kept in the queue/stack: " << algo_results[1] << endl <<
          "Number of nodes kept in the memory: " << cryptarithmetic_tree.get_nodes_in_memory() << endl <<
          "Running time: " << duration.count() / 1000000.0 << " seconds" << endl <<
          "Solution: " << cryptarithmetic_tree.get_solution() << endl;

  delete(algo_results);
  return 0;
}

Node::Node(int dla) {
  this->distinct_letter_amount = dla;
  this->parent = nullptr;
  this->index = 0;
  this->is_leaf = false;
  this->children = vector<Node*>(10);
  this->data = vector<vector<uint8_t>>(dla);

  // Zero out the data matrix
  for (int i = 0; i < dla; i++) {
    data[i] = vector<uint8_t>(10, 0);
  }
}

Node::Node(int dla, Node* parent, int row, int col) {
  this->distinct_letter_amount = dla;
  this->parent = parent;
  this->index = parent->index * 10 + col + 1;
  this->is_leaf = (row == dla - 1);
  this->children = vector<Node*>(10);
  this->data = this->parent->data;
  this->data[row][col] = 1;
}

vector<Node *> Node::get_children() const {
  return children;
}

void Node::set_children(vector<Node*> c) {
  this->children = c;
}

vector<vector<uint8_t>> Node::get_data() {
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
        cout << unsigned(data[i][j]) << "\t";
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

Tree::Tree(const string &o1, const string &o2, const string &r) {
  this->operand1 = o1;
  this->operand2 = o2;
  this->result = r;
  this->solution = "Solution not found!";
  this->solution_to_file = "Solution not found!\n";
  this->letters = get_distinct_letters();

  this->node_amount = 0;
  for (int i = 0; i < letters.length() + 1; i++) {
    this->node_amount += pow(10, i);
  }

  // Map the operands
  this->operand1_map = map_operand(operand1);
  this->operand2_map = map_operand(operand2);
  this->result_map = map_operand(result);

  // Get distinct letter amount
  int dla = letters.length();
  this->root = new Node(dla);
  this->nodes_in_memory = 1;

  // Fill the tree
  populate(0, root);
}

// Get distinct letters of the operands
string Tree::get_distinct_letters() {
  // Concatenate operands and the result
  string concat = operand1 + operand2 + result;
  transform(concat.begin(), concat.end(), concat.begin(), ::toupper); // Uppercase
  // Find distinct letters
  string distinct_letters;
  for (char i : concat) {
    if (distinct_letters.find(i) == string::npos) {
      distinct_letters += i;
    }
  }
  return distinct_letters;
}

// Map operands to the letters
vector<int> Tree::map_operand(string op) {
  vector<int> op_map = vector<int>(op.length());
  for (int i = 0; i < op.length(); i++) {
    for (int j = 0; j < letters.length(); j++) {
      if (op[i] == letters[j])
        op_map[i] = j;
    }
  }
  return op_map;
}

// Populate the tree with all possible permutations
void Tree::populate(int level, Node *parent) {
  if (level < letters.length()) {
    vector<Node *> children;
    // Create all ten children and push them
    for (int i = 0; i < 10; i++) {
      Node *child = new Node(letters.length(), parent, level, i);
      nodes_in_memory++;
      children.push_back(child);
    }
    parent->set_children(children);
    for (int i = 0; i < 10; i++) {
      // Create the whole tree only if there are less than 7 characters
      if (letters.length() > 6){
        if (satisfies_constraints(*children[i])){
          populate(level + 1, children[i]);
        }
      } else {
        populate(level + 1, children[i]);
      }
    }
  }
}

// Check if the node satisfies constraints of a possible solution
bool Tree::satisfies_constraints(Node &node) {
  // Make a copy of the data matrix
  vector<vector<uint8_t>> data = node.get_data();

  // constraint 1 - first letters of operands are not zero
  if (data[operand1_map[0]][0] || data[operand2_map[0]][0] || data[result_map[0]][0])
    return false;

  // constraint 2 - different letters are mapped to different numbers
  for (int i = 0; i < 10; i++) {
    int sum = 0;
    for (int j = 0; j < letters.length(); j++) {
      sum += data[j][i];
      if (sum > 1) {
        return false;
      }
    }
  }
  // Constraints satisfied!
  return true;
}

bool Tree::check_solution(Node &node) {
  // Make a copy of the data matrix
  vector<vector<uint8_t>> data = node.get_data();
  // Make sure a number is assigned to every letter (node is leaf)
  if (!node.leaf()) {
    return false;
  }

  // Get the values of the letters of the node
  vector<int> letter_values = vector<int>(letters.length());
  for (int i = 0; i < letters.length(); i++) {
    int j = 0;
    while (j < 10) {
      if (data[i][j] == 1) {
        letter_values[i] = j;
        j = 9;  // I could use break here but what the heck
      }
      j++;
    }
  }

  // Get the operand 1 value
  int operand1_value = 0;
  for (int i = 0; i < operand1.length(); i++) {
    operand1_value += letter_values[operand1_map[i]] * int(pow(10, operand1.length()-i-1));
  }
  // Get the operand 2 value
  int operand2_value = 0;
  for (int i = 0; i < operand2.length(); i++) {
    operand2_value += letter_values[operand2_map[i]] * int(pow(10, operand2.length()-i-1));
  }
  // Get the result's value
  int result_value = 0;
  for (int i = 0; i < result.length(); i++) {
    result_value += letter_values[result_map[i]] * int(pow(10, result.length()-i-1));
  }

  if (operand1_value + operand2_value == result_value) {
    // Construct the solution string
    solution = "";
    for (int i = 0; i < letters.length(); i++) {
      if (i+1 == letters.length())
        solution += string(1, letters[i]) + ": " + to_string(letter_values[i]);
      else
        solution += string(1, letters[i]) + ": " + to_string(letter_values[i]) + ", ";
    }

    // Construct the solution to the file string
    solution_to_file = "\t";
    for (int i = 0; i < 10; i++)
      solution_to_file += to_string(i) + "\t";
    solution_to_file += "\n";
    for (int i = 0; i < letters.length(); i++) {
      solution_to_file += string(1, tolower(letters[i])) + "\t";
      for (int j = 0; j < 10; j++) {
        if (j == 9) {
          if (data[i][j])
            solution_to_file += "1\n";
          else
            solution_to_file += ".\n";
        } else if (data[i][j]) {
          solution_to_file += "1\t";
        } else {
          solution_to_file += ".\t";
        }
      }
    }
    // Remove the new line at the end
    solution_to_file.pop_back();

    return true;
  } else {
    return false;
  }
}

int* Tree::bfs() {
  vector<Node *> q;
  vector<bool> visited = vector<bool>(node_amount, false);
  q.push_back(root);
  visited[root->get_index()] = true;
  int visited_nodes = 1;
  int queue_size = 1;
  int max_queue_size = queue_size;
  bool solution_found = false;

  while (!q.empty() && !solution_found) {
    if (queue_size > max_queue_size)
      max_queue_size = queue_size;

    Node *node = q.front();
    q.erase(q.begin());
    queue_size--;

    if (node->leaf()) {
      solution_found = check_solution(*node);
    } else {
      // Add children to the queue that satisfies the constraints
      vector<Node *> children = node->get_children();
      for (int i = 0; i < 10; i++) {
        if (!visited[children[i]->get_index()] && satisfies_constraints(*children[i])) {
          q.push_back(children[i]);
          queue_size++;
          visited[children[i]->get_index()] = true;
          visited_nodes++;
        }
      }
    }
  }
  int* to_return = new int[2];
  to_return[0] = visited_nodes;
  to_return[1] = max_queue_size;
  return to_return;
}

int* Tree::dfs() {
  vector<Node *> s;
  vector<bool> visited = vector<bool>(node_amount, false);
  s.push_back(root);
  visited[root->get_index()] = true;
  int visited_nodes = 1;
  int stack_size = 1;
  int max_stack_size = stack_size;
  bool solution_found = false;

  while (!s.empty() && !solution_found) {
    if (stack_size > max_stack_size)
      max_stack_size = stack_size;

    Node *node = s.back();
    s.pop_back();
    stack_size--;

    if (node->leaf()) {
      solution_found = check_solution(*node);
    } else {
      // Add children to the stack that satisfies the constraints
      vector<Node *> children = node->get_children();
      for (int i = 0; i < 10; i++) {
        if (!visited[children[i]->get_index()] && satisfies_constraints(*children[i])) {
          s.push_back(children[i]);
          stack_size++;
          visited[children[i]->get_index()] = true;
          visited_nodes++;
        }
      }
    }
  }
  int* to_return = new int[2];
  to_return[0] = visited_nodes;
  to_return[1] = max_stack_size;
  return to_return;
}

int Tree::get_nodes_in_memory() const {
  return nodes_in_memory;
}

string Tree::get_solution() const {
  return solution;
}

string Tree::get_solution_to_file() const {
  return solution_to_file;
}