#include "tree.h"
#include <bits/stdc++.h>
#include <cmath>

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

int Tree::bfs() {
  vector<Node *> q;
  q.push_back(root);
  int visited_nodes = 1;
  bool solution_found = false;

  while (!q.empty() && !solution_found) {
    Node *node = q.front();
    q.erase(q.begin());

    if (node->leaf()) {
        solution_found = check_solution(*node);
    } else {
      // Add children to the queue that satisfies the constraints
      vector<Node *> children = node->get_children();
      for (int i = 0; i < 10; i++) {
        if (satisfies_constraints(*children[i])) {
          q.push_back(children[i]);
          visited_nodes++;
        }
      }
    }
  }

  return visited_nodes;
}

int Tree::dfs() {
  vector<Node *> s;
  s.push_back(root);
  int visited_nodes = 1;
  bool solution_found = false;

  while (!s.empty() && !solution_found) {
    Node *node = s.back();
    s.pop_back();

    if (node->leaf()) {
        solution_found = check_solution(*node);
    } else {
      // Add children to the stack that satisfies the constraints
      vector<Node *> children = node->get_children();
      for (int i = 0; i < 10; i++) {
        if (satisfies_constraints(*children[i])) {
          s.push_back(children[i]);
          visited_nodes++;
        }
      }
    }
  }

  return visited_nodes;
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
