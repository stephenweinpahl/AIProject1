#Stephen Weinpahl and Michael Trinh
#Intro to Artificial Intellgence
#Project 1
#7/3/21

from os import startfile
import matplotlib.pyplot as plt
import matplotlib

# creates a maze object from an array that contains the state of the maze at a given row, col from a file
# states are: 0 -> open space, 1 -> barrier, 2-> failed path, 3 -> successfull path
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
        # create map that maps board state value to color
        # 0 -> white (open space)
        # 1 -> black (barrier)
        # 2 -> red (failed path)
        # 3 -> green (successful path)
        c_map = ["white", "black", "red", "green"]
        cm = matplotlib.colors.ListedColormap(c_map)

        # plot the board as a color image, vmax scales the color map to the largest integer
        # vmax = 3 scales the largest integer in the board to the last color in the colormap (green)
        plt.imshow(self.board, vmin = 0, vmax = 3, cmap=cm)
        plt.show()

#algorithm zero
def algoZero(board, start, goal):
    print("algo zero")

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
    maze1.visualize()
    
if __name__ == "__main__":
    main()

