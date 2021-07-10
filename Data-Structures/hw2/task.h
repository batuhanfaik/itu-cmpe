/* @Author
Student Name: Batuhan Faik Derinbay
Student ID : 150180705
Date: 31.10.19 */

/*
PLEASE DO NOT CHANGE THIS FILE 
*/

#define NAME_LENGTH 2

struct Task{
	char *name;
	int day;
	int time;
	int priority;

	Task *previous;
	Task *next;
	Task *counterpart;
};
