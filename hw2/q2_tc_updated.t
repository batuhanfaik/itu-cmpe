- init:
    run: rm -f q2.o q2
    blocker: true

- build:
    run: g++ -std=c++11 -Wall -Werror  q2.cpp -o q2  # timeout: 8
    blocker: true

- case1:
    run: ./q2
    points: 25
    script:
        - send: "path_info_1.txt"
        - expect: "Ma S1 R1 R2 Mo 19\r\n"           # timeout: 8
        - expect: _EOF_                             # timeout: 8
    return: 0
