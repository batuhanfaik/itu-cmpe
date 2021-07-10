- init:
    run: rm -f q1.o q1
    blocker: true

- build:
    run: g++ -std=c++11 -Wall -Werror  q1.cpp -o q1                   # timeout: 8
    blocker: true

- case0: 
    run: ./q1
    points: 25
    script:
        - send: "city_plan_1.txt"
        - expect: "Hipp Ch1 1\r\nGP Ch2 2\r\nHp3 Bas2 3\r\nCh2 Bas3 4\r\nGP Bas2 6\r\nHipp Hp1 7\r\nGP Hp4 8\r\nHipp Hp2 10\r\nGP Hipp 12\r\nBas1 Ch2 15\r\n68\r\n"           # timeout: 8
        - expect: _EOF_                                              # timeout: 8
    return: 0

- case1:
    run: ./q1
    points: 25
    script:
        - send: "city_plan_2.txt"
        - expect: "GP Hp1 1\r\nCh1 Hp2 3\r\nGP Ch1 9\r\nGP Ch2 10\r\nGP Hipp 15\r\n38\r\n"           # timeout: 8
        - expect: _EOF_                                              # timeout: 8
    return: 0
