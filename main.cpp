/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * /
* @ Filename: main
* @ Date: 31-Mar-2021
* @ AUTHOR: batuhanfaik
* @ Copyright (C) 2021 Batuhan Faik Derinbay
* @ Project: hw1
* @ Description: Cryptarithmetic, to run: ./hw1 DFS TWO TWO FOUR outputFileName
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>   // Required to measure time

#include "tree.h"

using namespace std;

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

  int visited_nodes;
  // Get the current time
  auto start_time = chrono::high_resolution_clock::now();

  // Create the tree
  Tree cryptarithmetic_tree = Tree(operand1, operand2, result);

  // Run the search
  if (algorithm == "DFS")
    visited_nodes = cryptarithmetic_tree.dfs();
  else
    visited_nodes = cryptarithmetic_tree.bfs();

  // Searching ends
  auto stop_time = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::microseconds>(stop_time - start_time);

  // Write to file
  ofstream file;
  file.open(output_file_name + ".txt");
  file << cryptarithmetic_tree.get_solution_to_file();
  file.close();

  cout << "Algorithm: " << algorithm << endl <<
          "Number of visited nodes: " << visited_nodes << endl <<
          "Maximum number of nodes kept in the memory: " << cryptarithmetic_tree.get_nodes_in_memory() << endl <<
          "Running time: " << duration.count() / 1000000.0 << " seconds" << endl <<
          "Solution: " << cryptarithmetic_tree.get_solution() << endl;

  return 0;
}