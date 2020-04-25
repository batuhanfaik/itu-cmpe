/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * /
* @ Filename: hw2.c
* @ Date: 21-Apr-2020
* @ AUTHOR: Batuhan Faik Derinbay
* @ Student ID: 150180705
* @ Copyright (C) 2020 Batuhan Faik Derinbay
* @ Project: BLG312E Homework 2
* @ Development Environment: Ubuntu 18.04, GDB 8.3, C Standard 99
* @ Description: Calculate prime numbers given a range of two integers
* @ Instructions:
*      To compile:     gcc hw2.c -o hw2 -std=c99 -pthread
*      To run:         ./hw2 interval_min interval_max np nt
*      Example:        ./hw2 101 200 2 2
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <pthread.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>

struct ThreadInfo {
    int process_no;
    int thread_no;
    int interval_start;
    int interval_end;
    int *shared_memory;
    int memory_offset;
};

void *thread_func(void *thread_struct) {
    struct ThreadInfo *t_info = (struct ThreadInfo *) thread_struct;

    printf("Thread %d.%d: ", t_info->process_no, t_info->thread_no);
    printf("Searching between %d-%d\n", t_info->interval_start, t_info->interval_end);

    // Find prime numbers
    int lower_bound, upper_bound, not_prime;

    lower_bound = t_info->interval_start;
    upper_bound = t_info->interval_end;
    upper_bound++;  // This fixes the non-inclusive upper boundary problem of the algorithm

    while (lower_bound < upper_bound) {
        not_prime = 0;

        // If low is a non-prime number, set the flag
        for (int i = 2; i <= lower_bound / 2; ++i) {
            if (lower_bound % i == 0) {
                not_prime = 1;
                break;
            }
        }

        // Save the prime number in the memory, if not prime, fill the memory space with 0
        if (not_prime == 0) {
            t_info->shared_memory[lower_bound - t_info->memory_offset] = lower_bound;
        } else {
            t_info->shared_memory[lower_bound - t_info->memory_offset] = 0;
        }
//            printf("%d ", lower_bound);
        ++lower_bound;
    }

    return NULL;
}

int main(int argc, char **argv) {
    // Master starts
    printf("Master: Started.\n");

    // Get command line arguments and convert to integer
    char *arg1 = argv[1];
    char *arg2 = argv[2];
    char *arg3 = argv[3];
    char *arg4 = argv[4];
    int INTERVAL_START = atoi(arg1);
    int INTERVAL_END = atoi(arg2);
    int PROCESS_AMOUNT = atoi(arg3);
    int THREAD_AMOUNT = atoi(arg4);


    // Calculate ranges and distribute
    int total_range = INTERVAL_END - INTERVAL_START + 1;
    int range_per_process = total_range / PROCESS_AMOUNT;
    int process_intervals[PROCESS_AMOUNT * 2];
    int interval_offset = 0;
    int residue;

    for (int i = 0; i < PROCESS_AMOUNT; i++) {
        if (i < total_range % PROCESS_AMOUNT) {
            interval_offset++;
            residue = 1;
        } else
            residue = 0;

        process_intervals[i * 2] = INTERVAL_START + (i * range_per_process) + interval_offset - residue;
        process_intervals[i * 2 + 1] = process_intervals[i * 2] + range_per_process - 1 + residue;

        // Print the number ranges that are being processed by the slave
//        printf("%d %d, ", process_intervals[i * 2], process_intervals[i * 2 + 1]);
        // Print how many numbers are being processed by the slave
//        printf("%d, ", process_intervals[i * 2 + 1] - process_intervals[i * 2] + 1);
    }

    // Handle IPC Keys and create shared memory
    char cwd[256];
    getcwd(cwd, 256);
    char *key_string = malloc(strlen(cwd) + 1);
    strcpy(key_string, cwd);
    int MEM_KEY = ftok(key_string, 1);
    free(key_string);

    int shmid = shmget(MEM_KEY, total_range * sizeof(int), 0666 | IPC_CREAT);
    if (shmid < 0) {
        perror("Unable to allocate memory!\n");
        exit(1);
    }

    // Attach allocated memory
    int *shared_memory = (int *) shmat(shmid, 0, 0);

    // Create NP slave processes
    for (int i = 0; i < PROCESS_AMOUNT; ++i) {
        int process_interval_start = process_intervals[i * 2];
        int process_interval_end = process_intervals[i * 2 + 1];
        // Check for faulty process interval
        if (process_interval_end < process_interval_start) {
            printf("Slave %d: Not required!\n", i + 1);
            _exit(0);
        }
        if (fork() == 0) {
            // Slave process starts

            printf("Slave %d: Started. ", i + 1);
            printf("Interval %d-%d\n", process_interval_start, process_interval_end);

            // Calculate ranges and distribute
            int process_total_range = process_interval_end - process_interval_start + 1;
            int range_per_thread = process_total_range / THREAD_AMOUNT;
            int thread_intervals[THREAD_AMOUNT * 2];
            int thread_interval_offset = 0;
            int residue_;

            for (int k = 0; k < THREAD_AMOUNT; k++) {
                if (k < process_total_range % THREAD_AMOUNT) {
                    thread_interval_offset++;
                    residue_ = 1;
                } else
                    residue_ = 0;

                thread_intervals[k * 2] =
                        process_interval_start + (k * range_per_thread) + thread_interval_offset - residue_;
                thread_intervals[k * 2 + 1] = thread_intervals[k * 2] + range_per_thread - 1 + residue_;

                // Print the number ranges that are being processed by the slave
//                printf("%d %d, ", thread_intervals[k * 2], thread_intervals[k * 2 + 1]);
                // Print how many numbers are being processed by the slave
//                printf("%d, ", thread_intervals[k * 2 + 1] - thread_intervals[k * 2] + 1);
            }

            // Create threads
            pthread_t thread_ids[THREAD_AMOUNT];
            struct ThreadInfo t_info[THREAD_AMOUNT];
            for (int j = 0; j < THREAD_AMOUNT; ++j) {
                pthread_attr_t attr;
                pthread_attr_init(&attr);

                t_info[j].process_no = i + 1;
                t_info[j].thread_no = j + 1;
                t_info[j].interval_start = thread_intervals[j * 2];
                t_info[j].interval_end = thread_intervals[j * 2 + 1];
                t_info[j].shared_memory = shared_memory;
                t_info[j].memory_offset = INTERVAL_START;

                // Check for faulty thread interval
                if (t_info[j].interval_end < t_info[j].interval_start) {
                    printf("Thread %d.%d: Not required!\n", t_info[j].process_no, t_info[j].thread_no);
                } else {
                    pthread_create(&thread_ids[j], &attr, thread_func, &t_info[j]);
                }
            }

            // Wait until all threads are done
            for (int j = 0; j < THREAD_AMOUNT; ++j) {
                pthread_join(thread_ids[j], NULL);
            }

            // Process completed
            printf("Slave %d: Done.\n", i + 1);
            _exit(0);
        }
    }

    // Wait for slave processes to complete
    for (int j = 0; j < PROCESS_AMOUNT; ++j) {
        wait(NULL);
    }

    // Print prime numbers
    printf("Prime numbers: ");
    for (int i = 0; i < total_range; ++i) {
        if (shared_memory[i])
            printf("%d ", shared_memory[i]);
    }

    // Detach shared memory and remove it
    shmdt(shared_memory);
    shmctl(shmid, IPC_RMID, 0);

    // Master has finished
    printf("\nMaster: Done.\n");

    return 0;
}
