/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * /
* @ Filename: hw3.c
* @ Date: 03-Jun-2020
* @ AUTHOR: Batuhan Faik Derinbay
* @ Student ID: 150180705
* @ Copyright (C) 2020 Batuhan Faik Derinbay
* @ Project: BLG312E Homework 3
* @ Development Environment: Ubuntu 18.04, GDB 8.3, C Standard 99
* @ Description: The Moneybox Homework
* @ Instructions:
*      To compile:     gcc hw3.c -o hw3 -std=c99
*      To run:         ./hw3 N Ni Nd ti td
*      Example:        ./hw3 150 4 2 2 4
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#define _POSIX_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <sys/types.h>
#include <signal.h>
#include <sys/shm.h>
#include <unistd.h>

// Semaphore lock
int sem_lock(int sem_id, short val) {
    struct sembuf semaphore;
    semaphore.sem_num = 0;
    semaphore.sem_op = (short) (-1 * val);
    semaphore.sem_flg = 1;
    return semop(sem_id, &semaphore, 1);
}

// Semaphore unlock
int sem_unlock(int sem_id, short val) {
    struct sembuf semaphore;
    semaphore.sem_num = 0;
    semaphore.sem_op = val;
    semaphore.sem_flg = 1;
    return semop(sem_id, &semaphore, 1);
}

void step_fibonacci(int *vec) {
    int tmp = vec[0] + vec[1];
    vec[0] = vec[1];
    vec[1] = tmp;
}

struct ProcessInfo {
    int process_type;
    int process_task_idx;
    int *fibonacci_vector;
    int fibonacci_idx;
};

int main(int argc, char **argv) {
    // Variable initializations
    int i, tmp, money_thresh, increaser_turn, decreaser_turn;
    short increaser_amount, decreaser_amount;
    int money_id, inc_turn_id, dec_turn_id, pass_flag_id;   // Shared memories
    int *money, *inc_turn, *dec_turn, *pass_flag;    // Memory pointers
    int master_sync, money_mutex, inc_turn_sync, dec_turn_sync, inc_wait, dec_wait;   // Semaphores
    int inc_count, dec_count;   // Counting semaphores
    int total_processes_idx, increaser_processes_idx, decreaser_processes_idx;  // Process counters

    pid_t pid;
    pid_t *pid_arr;

    // Check for faulty argument inputs
    if (argc > 6) {
        printf("\nToo many operands!\n");
        exit(EXIT_FAILURE);
    } else if (argc < 6) {
        printf("\nMissing operands!\n");
        exit(EXIT_FAILURE);
    }

    money_thresh = (short) atoi(argv[1]);
    increaser_amount = (short) atoi(argv[2]);
    decreaser_amount = (short) atoi(argv[3]);
    increaser_turn = (short) atoi(argv[4]);
    decreaser_turn = (short) atoi(argv[5]);

    // Handle IPC Keys, create shared memory for semaphores and variables
    char cwd[256];
    getcwd(cwd, 256);
    char *key_string = malloc(strlen(cwd) + 1);
    strcpy(key_string, cwd);
    int MEM_KEY = ftok(key_string, 1);
    free(key_string);

    // Shared memory declarations for variables
    money_id = shmget(MEM_KEY, sizeof(int), 0644 | IPC_CREAT);
    money = shmat(money_id, NULL, 0);
    *money = 0;

    pass_flag_id = shmget(MEM_KEY + 1, sizeof(int), 0644 | IPC_CREAT);
    pass_flag = shmat(pass_flag_id, NULL, 0);
    *pass_flag = 0;

    inc_turn_id = shmget(MEM_KEY + 2, sizeof(int), 0644 | IPC_CREAT);
    inc_turn = shmat(inc_turn_id, NULL, 0);
    *inc_turn = 0;

    dec_turn_id = shmget(MEM_KEY + 3, sizeof(int), 0644 | IPC_CREAT);
    dec_turn = shmat(dec_turn_id, NULL, 0);
    *dec_turn = 0;

    // Shared memory declarations for semaphores
    master_sync = semget(MEM_KEY + 4, 1, 0700 | IPC_CREAT);
    semctl(master_sync, 0, SETVAL, 0);

    money_mutex = semget(MEM_KEY + 5, 1, 0700 | IPC_CREAT);
    semctl(money_mutex, 0, SETVAL, 1);

    inc_turn_sync = semget(MEM_KEY + 6, 1, 0700 | IPC_CREAT);
    semctl(inc_turn_sync, 0, SETVAL, increaser_amount);

    dec_turn_sync = semget(MEM_KEY + 7, 1, 0700 | IPC_CREAT);
    semctl(dec_turn_sync, 0, SETVAL, 0);

    inc_count = semget(MEM_KEY + 8, 1, 0700 | IPC_CREAT);
    semctl(inc_count, 0, SETVAL, 0);

    dec_count = semget(MEM_KEY + 9, 1, 0700 | IPC_CREAT);
    semctl(dec_count, 0, SETVAL, 0);

    inc_wait = semget(MEM_KEY + 10, 1, 0700 | IPC_CREAT);
    semctl(inc_wait, 0, SETVAL, 0);

    dec_wait = semget(MEM_KEY + 11, 1, 0700 | IPC_CREAT);
    semctl(dec_wait, 0, SETVAL, 0);

    // Create a PID array to store all PIDs, so they can be killed later
    pid_arr = (pid_t *) malloc(sizeof(pid_t) * (increaser_amount + decreaser_amount));

    // Indexes for the PID array
    total_processes_idx = 0;
    increaser_processes_idx = 0;
    decreaser_processes_idx = 0;

    struct ProcessInfo p_info;
    p_info.process_type = -1;
    p_info.process_task_idx = -1;
    p_info.fibonacci_vector = NULL;

    // Print the current money
    printf("Master Process: Current money is %d\n", *money);

    // Create increment by 10 processes
    for (i = 0; i < increaser_amount / 2; i++) {
        pid = fork();
        if (pid == 0) {     // Setup process specific variables
            p_info.process_type = 0;
            p_info.process_task_idx = increaser_processes_idx;
            break;
        } else {    // Increment PID array indexes
            total_processes_idx += 1;
            increaser_processes_idx += 1;
            pid_arr[total_processes_idx - 1] = pid;
        }
    }
    // Create increment by 15 processes
    for (i = 0; i < increaser_amount / 2 && p_info.process_type == -1; i++) { // Increment by 15
        pid = fork();
        if (pid == 0) {     // Setup process specific variables
            p_info.process_type = 1;
            p_info.process_task_idx = increaser_processes_idx;
            break;
        } else {    // Increment PID array indexes
            total_processes_idx += 1;
            increaser_processes_idx += 1;
            pid_arr[total_processes_idx - 1] = pid;
        }
    }
    // Create odd decreasing processes
    for (i = 0; i < decreaser_amount / 2 && p_info.process_type == -1; i++) {
        pid = fork();
        if (pid == 0) {     // Setup process specific variables
            p_info.process_type = 2;
            p_info.process_task_idx = decreaser_processes_idx;
            break;
        } else {    // Increment PID array indexes
            total_processes_idx += 1;
            decreaser_processes_idx += 1;
            pid_arr[total_processes_idx - 1] = pid;
        }
    }
    // Create even decreasing processes
    for (i = 0; i < decreaser_amount / 2 && p_info.process_type == -1; i++) { // Decrement even
        pid = fork();
        if (pid == 0) {     // Setup process specific variables
            p_info.process_type = 3;
            p_info.process_task_idx = decreaser_processes_idx;
            break;
        } else {    // Increment PID array indexes
            total_processes_idx += 1;
            decreaser_processes_idx += 1;
            pid_arr[total_processes_idx - 1] = pid;
        }
    }

    // For decreasing processes, create a fibonacci vector
    if (p_info.process_type == 2 || p_info.process_type == 3) {
        p_info.fibonacci_vector = (int *) malloc(sizeof(int) * 2);
        p_info.fibonacci_vector[0] = 0;
        p_info.fibonacci_vector[1] = 1;
        p_info.fibonacci_idx = 1;
    }

    // Start children processes turns
    if (p_info.process_type != -1) {
        while (semctl(master_sync, 0, GETVAL) == 0) {
            // If the process is increment by 10
            if (p_info.process_type == 0) {
                sem_lock(inc_turn_sync, 1);
                sem_lock(money_mutex, 1);
                *money += 10;
                printf("Increaser Process %d: Current money is %d", p_info.process_task_idx, *money);
                sem_unlock(inc_count, 1);
                // If all increaser processes completed their turn
                if (semctl(inc_count, 0, GETVAL) == increaser_amount) {
                    *inc_turn += 1;
                    printf(", increaser processes finished their turn %d\n", *inc_turn);
                    semctl(inc_count, 0, SETVAL, 0);
                    if (*money >= money_thresh && *pass_flag == 0) {
                        *pass_flag = 1;
                    }
                    if (*inc_turn % increaser_turn == 0 && *pass_flag) {
                        sem_unlock(dec_turn_sync, decreaser_amount);
                    } else {
                        sem_unlock(inc_turn_sync, increaser_amount);
                    }
                    sem_unlock(inc_wait, increaser_amount);
                } else {
                    printf("\n");
                }
                sem_unlock(money_mutex, 1);
                sem_lock(inc_wait, 1);
                // If the process is increment by 15
            } else if (p_info.process_type == 1) {
                sem_lock(inc_turn_sync, 1);
                sem_lock(money_mutex, 1);
                *money += 15;
                printf("Increaser Process %d: Current money is %d", p_info.process_task_idx, *money);
                sem_unlock(inc_count, 1);
                // If all increaser processes completed their turn
                if (semctl(inc_count, 0, GETVAL) == increaser_amount) {
                    *inc_turn += 1;
                    printf(", increaser processes finished their turn %d\n", *inc_turn);
                    semctl(inc_count, 0, SETVAL, 0);
                    if (*money >= money_thresh && *pass_flag == 0) {
                        *pass_flag = 1;
                    }
                    if (*inc_turn % increaser_turn == 0 && *pass_flag) {
                        sem_unlock(dec_turn_sync, decreaser_amount);
                    } else {
                        sem_unlock(inc_turn_sync, increaser_amount);
                    }
                    sem_unlock(inc_wait, increaser_amount);
                } else {
                    printf("\n");
                }
                sem_unlock(money_mutex, 1);
                sem_lock(inc_wait, 1);
                // If the process is an odd decrementer
            } else if (p_info.process_type == 2) {
                sem_lock(dec_turn_sync, 1);
                sem_lock(money_mutex, 1);
                // If the money is odd, decrement
                if (*money % 2 == 1) {
                    tmp = *money - p_info.fibonacci_vector[1];
                    if (tmp < 0) {  // Decreaser is greater than the money, kill all
                        printf("Decreaser Process %d: Current money is less than %d, signaling master to finish (%dth fibonacci number for decreaser %d)\n",
                               p_info.process_task_idx, p_info.fibonacci_vector[1], p_info.fibonacci_idx,
                               p_info.process_task_idx);
                        sem_unlock(master_sync, 1);
                    } else {    // Decrease the money
                        *money = tmp;
                        printf("Decreaser Process %d: Current money is %d (%d. fibonacci number for decreaser %d)",
                               p_info.process_task_idx, *money, p_info.fibonacci_idx, p_info.process_task_idx);
                        sem_unlock(dec_count, 1);
                        // If all decreaser processes completed their turn
                        if (semctl(dec_count, 0, GETVAL) == decreaser_amount) {
                            *dec_turn += 1;
                            printf(", decreaser processes finished their turn %d\n", *dec_turn);
                            semctl(dec_count, 0, SETVAL, 0);
                            if (*dec_turn % decreaser_turn == 0) {
                                sem_unlock(inc_turn_sync, increaser_amount);
                            } else {
                                sem_unlock(dec_turn_sync, decreaser_amount);
                            }
                            sem_unlock(dec_wait, decreaser_amount);
                        } else {
                            printf("\n");
                        }
                    }
                    step_fibonacci(p_info.fibonacci_vector);
                    p_info.fibonacci_idx += 1;
                } else {    // If the money is even, send sync signals and pass
                    sem_unlock(dec_count, 1);
                    if (semctl(dec_count, 0, GETVAL) == decreaser_amount) {
                        *dec_turn += 1;
                        semctl(dec_count, 0, SETVAL, 0);
                        if (*dec_turn % decreaser_turn == 0) {
                            sem_unlock(inc_turn_sync, increaser_amount);
                        } else {
                            sem_unlock(dec_turn_sync, decreaser_amount);
                        }
                        sem_unlock(dec_wait, decreaser_amount);
                    }
                }
                sem_unlock(money_mutex, 1);
                sem_lock(dec_wait, 1);
                // If the process is even decrementer
            } else {
                sem_lock(dec_turn_sync, 1);
                sem_lock(money_mutex, 1);
                // If the money is even, decrement
                if (*money % 2 == 0) {
                    tmp = *money - p_info.fibonacci_vector[1];
                    if (tmp < 0) {  // Decreaser is greater than the money, kill all
                        printf("Decreaser Process %d: Current money is less than %d, signaling master to finish (%dth fibonacci number for decreaser %d)\n",
                               p_info.process_task_idx, p_info.fibonacci_vector[1], p_info.fibonacci_idx,
                               p_info.process_task_idx);
                        sem_unlock(master_sync, 1);
                    } else {    // Decrease the money
                        *money = tmp;
                        printf("Decreaser Process %d: Current money is %d (%d. fibonacci number for decreaser %d)",
                               p_info.process_task_idx, *money, p_info.fibonacci_idx, p_info.process_task_idx);
                        sem_unlock(dec_count, 1);
                        // If all decreaser processes completed their turn
                        if (semctl(dec_count, 0, GETVAL) == decreaser_amount) {
                            *dec_turn += 1;
                            printf(", decreaser processes finished their turn %d\n", *dec_turn);
                            semctl(dec_count, 0, SETVAL, 0);
                            if (*dec_turn % decreaser_turn == 0) {
                                sem_unlock(inc_turn_sync, increaser_amount);
                            } else {
                                sem_unlock(dec_turn_sync, decreaser_amount);
                            }
                            sem_unlock(dec_wait, decreaser_amount);
                        } else {
                            printf("\n");
                        }
                    }
                    step_fibonacci(p_info.fibonacci_vector);
                    p_info.fibonacci_idx += 1;
                } else {    // If the money is odd, send sync signals and pass
                    sem_unlock(dec_count, 1);
                    if (semctl(dec_count, 0, GETVAL) == decreaser_amount) {
                        *dec_turn += 1;
                        semctl(dec_count, 0, SETVAL, 0);
                        if (*dec_turn % decreaser_turn == 0) {
                            sem_unlock(inc_turn_sync, increaser_amount);
                        } else {
                            sem_unlock(dec_turn_sync, decreaser_amount);
                        }
                        sem_unlock(dec_wait, decreaser_amount);
                    }
                }
                sem_unlock(money_mutex, 1);
                sem_lock(dec_wait, 1);
            }
        }
    }

    // Wait for all children processes to finish
    sem_lock(master_sync, 1);
    printf("Master Process: Killing all children and terminating the program\n");
    for (i = 0; i < total_processes_idx; i++) {
        kill(pid_arr[i], SIGTERM);
    }

    // Detach shared memories and remove them
    shmdt(money);
    shmctl(money_id, IPC_RMID, 0);
    shmdt(pass_flag);
    shmctl(pass_flag_id, IPC_RMID, 0);
    shmdt(inc_turn);
    shmctl(inc_turn_id, IPC_RMID, 0);
    shmdt(dec_turn);
    shmctl(dec_turn_id, IPC_RMID, 0);
    // Remove semaphores
    semctl(master_sync, 0, IPC_RMID);
    semctl(money_mutex, 0, IPC_RMID);
    semctl(inc_turn_sync, 0, IPC_RMID);
    semctl(dec_turn_sync, 0, IPC_RMID);
    semctl(inc_count, 0, IPC_RMID);
    semctl(dec_count, 0, IPC_RMID);
    semctl(inc_wait, 0, IPC_RMID);
    semctl(dec_wait, 0, IPC_RMID);
    // Delete previously allocated memory for PID array
    free(pid_arr);

    return 0;
}