import numpy as np
import dlib
import cv2
import pyautogui
import time

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/Users/ata/Desktop/vision/BLG453/term project/part2/shape_predictor_68_face_landmarks.dat')

def face_detect():
    img = pyautogui.screenshot(region=(1680, 0, 240, 240))  # face coord
    img = np.array(img)[:, :, :3].astype(np.uint8)
    rec = detector(img)
    try:
        pts = predictor(img,rec[0])
        x1, y1 = pts.part(37).x, pts.part(37).y
        x2, y2 = pts.part(19).x, pts.part(19).y
        dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 1 / 2
        return 1 if dist > 400 else 2
    except:
        return 0

def move(direction,step=2,s=0.37):
    for _ in range(step):
        pyautogui.keyDown(direction)
        time.sleep(s)
        pyautogui.keyUp(direction)

def move1(direction,step=2,s=0.37):
    pyautogui.keyDown('Shift')
    for _ in range(step):

        pyautogui.press(direction)
    pyautogui.keyUp('Shift')

def empty_check(direction):
    im = pyautogui.screenshot()
    im = np.array(im)[:, :, :3].astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    x=958        # canavarin karesi
    y = 300
    if direction == 'A':
        x-=270       #patlayabilir
    elif direction == 'D':
        x+=270
    elif direction == 'S':
        y+=270
    else:
        y-=270
    val = im[y,x]

    return False if (val > 230 )^( val <50) else True


def revert_direction(direction):
    if direction == 'A':
        return 'D'
    elif direction == 'D':
        return 'A'
    elif direction == 'W':
        return 'S'
    else :
        return 'W'
def check_move(direction,grid,ind):
    move(direction, step=1, s=0.17)
    time.sleep(0.5)
    face = face_detect()
    time.sleep(0.1)
    empty = empty_check(direction)
    print(face, empty)
    if face == 0:
        exit()
    if face == 1 and empty == False :
        # time.sleep(0.5)
        move(direction, step=1, s=0.57)
        time.sleep(0.5)
        return  True
    else :
        # time.sleep(0.5)
        r = revert_direction(direction)
        move(r, step=1, s=0.17)
        time.sleep(0.5)
        grid[ind[0],ind[1]]=2
        return False

time.sleep(2)
grid = np.zeros((100,100))# 1 for seen 2 for blocked
ind = [50,50]#x y
grid[50,50]= 1
stack = []
while True :
    prev_ind = ind
    if grid[ind[0],ind[1]+1]==0 and check_move('W',grid,[ind[0],ind[1]+1]):
        stack.append(ind)
        ind[1] += 1
        grid[ind[0],ind[1]] = 1
    elif grid[ind[0]+1,ind[1]]==0 and check_move('D',grid,[ind[0]+1,ind[1]]):
        stack.append(ind)
        ind[0] += 1
        grid[ind[0],ind[1]] = 1
    elif grid[ind[0]-1,ind[1]]==0 and check_move('A',grid,[ind[0]-1,ind[1]]):
        ind[0] -= 1
        grid[ind[0],ind[1]] = 1
        stack.append(ind)
    elif grid[ind[0],ind[1]-1]==0 and check_move('S',grid,[ind[0],ind[1]-1]):
        ind[1] -= 1
        grid[ind[0],ind[1]] = 1
        stack.append(ind)
    elif [ind[0],ind[1]+1]== [stack[-1][0],stack[-1][1]] and check_move('W',grid,[ind[0],ind[1]+1]):
        grid[ind[0],ind[1]] = 2
        stack.pop()
        ind[1] += 1
    elif [ind[0]+1,ind[1]]==[stack[-1][0],stack[-1][1]] and check_move('D',grid,[ind[0]+1,ind[1]]):
        grid[ind[0],ind[1]] = 2
        stack.pop()
        ind[0] += 1
    elif [ind[0]-1,ind[1]]==[stack[-1][0],stack[-1][1]] and check_move('A',grid,[ind[0]-1,ind[1]]):
        grid[ind[0],ind[1]] = 2
        stack.pop()
        ind[0] -= 1
    elif [ind[0],ind[1]-1]==[stack[-1][0],stack[-1][1]] and check_move('S',grid,[ind[0],ind[1]-1]):
        grid[ind[0],ind[1]] = 2
        stack.pop()
        ind[1] -= 1

#TODO 1. MERKEZI KALIBRE ET 2. BITIS NOKTASI ATA