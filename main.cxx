
#include <RandomWorld.hxx>
#include <RandomWorldConfig.hxx>
#include <Exception.hxx>
#include <Config.hxx>
#include <eigen3/Eigen/Dense>

#include <iostream>
#include <cstdlib>

Eigen::IOFormat OctaveFmt(3, 0, ", ", ";\n", "", "", "[", "]");

void printFloatMatrix(std::vector<Eigen::Triplet<float>> matrix, int width, int height, std::string name="M")
{
    SparseMatrixType sparseM(width, height);
    sparseM.setFromTriplets(matrix.begin(), matrix.end());
    std::cout << "\033[1;35m\n" << name << " matrix:" << "\033[0m" << std::endl;
    std::cout << Eigen::MatrixXf(sparseM).format(OctaveFmt) << std::endl;
}

void printIntMatrix(SparseMatrixType matrix, std::string name="M")
{   
    std::cout << "\033[1;35m\n" << name << " matrix:" << "\033[0m" << std::endl;
    std::cout << Eigen::MatrixXf(matrix).format(OctaveFmt) << std::endl;
}

int calculatePstar(std::vector<SparseMatrixType> states_rolls, std::vector<SparseMatrixType> reward_rolls, bool debug) 
{
    if (debug) 
    {
        for(std::vector<SparseMatrixType>::size_type i = 0; i != states_rolls.size(); i++) 
        {
            //TODO: WARNING!!! We loose sparsity here to print!!!!
            std::cout << "\033[1;34m\n" << "Roll " << i << "\033[0m" << std::endl;
            //printIntMatrix(states_rolls[i], "State");
            //printIntMatrix(reward_rolls[i], "Reward");
        }
    }
}

int updatePolicy(int pstar, int state) 
{
	std::cout << "Updating policy for state " << state << std::endl;
	return 0;
}

int main(int argc, char *argv[])
{
    int maxIt=2;
    int maxRolls=2;

    try
    {
        if(argc>2)
        {
            throw Engine::Exception("USAGE: randomWalkers [config file]");
        }

        std::string fileName("config.xml");
        if(argc!=1)
        {
            fileName = argv[1];
        }

        Examples::RandomWorldConfig * randomConfig =  new Examples::RandomWorldConfig(fileName);

        randomConfig->loadFile();
        int maxAgents = randomConfig->_numAgents;
        int maxSteps = randomConfig->getNumSteps();
        std::cout << "Creating matrices of (" << maxAgents << "x" << maxSteps << ")" << std::endl; 

        for(int i=0; i<maxIt; i++) 
        {
            std::cout << "\033[1;33m\n" << "ITERATION " << i << "\033[0m" << std::endl;
            
            std::vector<SparseMatrixType> state_rolls;
            std::vector<SparseMatrixType> reward_rolls;
            std::vector<SparseMatrixType> phi_rolls;

            // Execute tau rollouts
            for(int tau=0; tau<maxRolls; tau++) 
            {
                Examples::RandomWorld world(new Examples::RandomWorldConfig(fileName), world.useOpenMPSingleNode());

                world.initialize(argc, argv);
                world.initL();
                world.initBasis();
                int gridSize = world.getBoundaries()._size._width*world.getBoundaries()._size._height;
                //printFloatMatrix(world._L_spr_coeff, gridSize, gridSize, "L");
                world.run();

                SparseMatrixType states(maxAgents,maxSteps);
                states.setFromTriplets(world._pos_spr_coeff.begin(), world._pos_spr_coeff.end());
                state_rolls.push_back(states);

                SparseMatrixType rewards(maxAgents,maxSteps);
                rewards.setFromTriplets(world._rwd_spr_coeff.begin(), world._rwd_spr_coeff.end());
                reward_rolls.push_back(rewards);
            }

            // Calculate p* for each trajectory
            int pstar = calculatePstar(state_rolls, reward_rolls, true);

            // Update policy
            updatePolicy(pstar, i);
        }
    }
    catch( std::exception & exceptionThrown )
    {
        std::cout << "exception thrown: " << exceptionThrown.what() << std::endl;
    }
    return 0;
}
