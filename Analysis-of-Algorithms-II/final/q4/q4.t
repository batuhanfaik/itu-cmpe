- init:
    run: rm -f q4.o q4
    blocker: true

- build:
    run: g++ -std=c++11 -Wall -Werror  q4.cpp -o q4                   # timeout: 8
    blocker: true

- case1:
    run: ./q4
    points: 5
    script:
        - send: "q4_tc1.txt"
        - expect: "False\r\n"           # timeout: 8
        - expect: _EOF_                                              # timeout: 8
    return: 0

- case2:
    run: ./q4
    points: 5
    script:
        - send: "q4_tc2.txt"
        - expect: "True\r\n"           # timeout: 8
        - expect: _EOF_                                              # timeout: 8
    return: 0
