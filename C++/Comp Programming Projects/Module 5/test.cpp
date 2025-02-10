#include <iostream>
#include <vector>

class Maze {
    std::vector<std::vector<int>> maze;
    std::vector<int> from;
    std::vector<int> to;

    public: 
        int rows, cols, seed;
        void setup(int rows, int cols);
        bool visited(std::vector<int> from);
        bool findPath(std::vector<std::vector<int>> maze, std::vector<int> from, std::vector<int> to);


};

/**void Maze::setup(int rows, int cols) {
    int mazeSize = rows*cols;
    std::vector<std::vector<int>> maze (mazeSize, std::vector<int> (mazeSize, 0));
    for (int i; i < maze.size(); i++) {
        for (int j; j < maze.size(); j++){
            std::cout << maze.at(i).at(j);
        }
    }
}**/

bool Maze::visited(std::vector<int> from) {

}

bool Maze::findPath(std::vector<std::vector<int>> maze, std::vector<int> from, std::vector<int> to) {
    from = {0,0};

    if (from == to) {
        return true;
    }

}


int main(int argc, char **argv) {
    Maze m;
    try {
        if (argc == 4) {
            m.rows = std::stoi(argv[1]);
            m.cols = std::stoi(argv[2]);
            m.seed = std::stoi(argv[3]);
        }
        else if (argc == 3) {
            m.rows = std::stoi(argv[1]);
            m.cols = std::stoi(argv[2]);
        }
        else {
            throw std::runtime_error("invalid input");
        }
        m.setup(m.rows, m.cols);
        



    }
    catch (std::runtime_error& excpt) {
        std::cout << excpt.what() << std::endl;
    }


}