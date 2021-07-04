#Stephen Weinpahl and Michael Trinh
#Intro to Artificial Intellgence
#Project 1
#7/3/21
import time
from os import startfile
import matplotlib.pyplot as plt
import matplotlib
from collections import deque
import heapq

# creates a maze object from an array that contains the state of the maze at a given row, col from a file
# states are: 0 -> open space, 1 -> barrier, 2-> failed path, 3 -> successfull path, 4 -> start point, 5 -> end point
class maze:
    def __init__(self, file, start, goal, size):
        self.board = [[0 for i in range(size)] for j in range(size)]
        self.file = file
        self.start = start
        self.goal = goal
        self.size = size

    def buildMaze(self):
        file = self.file
        f = open(file, "r")
        lines = f.readlines()
        for line in lines:
            data = line.split()
            row = int(data[0])
            col = int(data[1])
            state = int(data[2])
            self.board[row][col] = state

    # creates a color map of a given board  
    def visualize(self):
        
        startX = self.start[0]
        startY = self.start[1]
        goalX = self.goal[0]
        goalY = self.goal[1]

        self.board[startX][startY] = 4
        self.board[goalX][goalY] = 5
        # create map that maps board state value to color
        # 0 -> white (open space)
        # 1 -> black (barrier)
        # 2 -> red (failed path)
        # 3 -> green (successful path)
        # 4 -> blue (start point)
        # 5 -> yellow (goal point)
        c_map = ["white", "black", "red", "green", "blue", "yellow"]
        cm = matplotlib.colors.ListedColormap(c_map)

        # plot the board as a color image, vmax scales the color map to the largest integer
        # vmax = 5 scales the largest integer in the board to the last color in the colormap (purple)
        plt.imshow(self.board, vmin = 0, vmax = 5, cmap=cm)
        plt.show()

    # Check to see whether a given cordinate is valid given the size of the maze
def isValid(row, col, size):
    return (row >= 0) and (col >= 0) and (row < size) and (col < size)

# Return list of valid neighbors from a starting point along with associated cost
def getNeighbors(row, col, board, currCost):
    # These arrays are used to get row and column
    # numbers of 4 neighbours of a given cell
    neighbors = []
    rowNum = [-1, 0, 0, 1]
    colNum = [0, -1, 1, 0]
    for i in range(4):
        x = row + rowNum[i]
        y = col + colNum[i]
        if isValid(x, y, len(board)) and board[x][y] != 1:
            neighbors.append([currCost + getCost(row, x), x, y])

    return neighbors

# Assign cost to a move from x1 to x2 (only x cordinates of two points)
# Moving up or down has cost = 2, left right has cost = 1
def getCost(x1, x2):
    #check to see if move is up or down
    if x1 != x2:
        return 2
    else:
        return 1

def findPath(maze, parent, start, goal):
    x = goal[0]
    y = goal[1]
    while x != start[0] or y != start[1]:
        tempX = parent[x][y][0]
        tempY = parent[x][y][1]
        x = tempX
        y = tempY
        maze.board[x][y] = 3

# algorithm zero: Uniform Cost Search to find shortest path from start to goal
def algoZero(maze):
    start = maze.start
    goal = maze.goal
    board = maze.board

    #check to see if start and end points are on the board
    if not isValid(start[0], start[1], len(board)) or not isValid(goal[0], goal[1], len(board)):
        return -1

    # check to see if start and end point are valid (not a barrier), if they are at barrier (1) return -1
    if board[start[0]][start[1]] != 0 or board[goal[0]][goal[1]] != 0:
        return -1

    # initialize a visited array set to false, when a cell is visited mark it as true and 
    # distTo array to determine minimum cost of getting to a point intitialized to max int value 
    # parent array to reconstruct the path if it exists
    visited = [[False for i in range(len(board[0]))] for j in range(len(board))]
    distTo = [[float('inf') for i in range(len(board[0]))] for j in range(len(board))]
    parent = [[None for i in range(len(board[0]))] for j in range(len(board))]
    distTo[start[0]][start[1]] = 0
    
    # create a priority queue using python deque class for BFS and push start point onto it 
    # queue contains a list of [cost, x, y]
    minQueue = []
    heapq.heappush(minQueue, [0, start[0], start[1]])
    while len(minQueue) > 0:
        # curr is the least costly item in the frontier
        curr = heapq.heappop(minQueue)
        cost = curr[0]
        x = curr[1]
        y = curr[2]
        maze.board[x][y] = 2

        # Check to see if current position is solution, if it is then reconstruct the path and return the cost
        if x == goal[0] and y == goal[1]:
            maze.board[x][y] = 5
            findPath(maze, parent, start, goal)
            return cost
        
        # only check cells that are not visited
        if visited[x][y] == False:
            frontier = getNeighbors(x, y, board, cost)
            if frontier:
                for cell in frontier:
                    # check if the current distance is larger than the recently calculated distance from the parent cell to the child cell, if not there is a faster route
                    i = cell[1]
                    j = cell[2]
                    childCost = cell[0]
                    if distTo[i][j] > childCost:
                        heapq.heappush(minQueue, cell)
                        distTo[i][j] = cell[0]
                        # add the new cell parent into the parent list
                        parent[i][j] = [x, y]
                        
        visited[x][y] = True
    return -1

#algorithm one
def algoOne(board, start, goal):
    print("algo one")

#algorithm two
def algoTwo(board, start, goal):
    print("algo zero")

#algorithm three
def algoThree(board, start, goal):
    print("algo three")

#algorithm four
def algoFour(board, start, goal):
    print("algo four")

# opens the problem.txt file and populates the neccessary information to create the maze object
# algo is the number of alrogithm that should be used to solve the maze
def main():
    file = "problem.txt"
    f = open(file, "r")
    lines = f.readlines()
    size = int(lines[0])
    start = lines[1].split()
    start[0] = int(start[0])
    start[1] = int(start[1])
    goal = lines[2].split()
    goal[0] = int(goal[0])
    goal[1] = int(goal[1])
    algo = int(lines[3])
    file = "maze_" + str(lines[4].strip()) + ".txt"
    maze1 = maze(file, start, goal, size)
    maze1.buildMaze()
    name = ["Uniform Cost Search Algorithm"]
    selector = [algoZero(maze1)]
    tic = time.perf_counter()
    result = selector[algo]
    toc = time.perf_counter()
    performance = toc - tic
    if result == -1:
        print("Error")
    else:
        print("The shortest path for maze_" + str(lines[4].strip()) + " using the " + name[algo] + " is " + str(result))
        print("It took " + str(performance) + " seconds to calculate the path")
    maze1.visualize()
if __name__ == "__main__":
    main()

