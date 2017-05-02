
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

int calculatePtheta(std::vector<SparseMatrixType> states_rolls, std::vector<SparseMatrixType> reward_rolls, bool debug) 
{
    std::cout << "Updating p_theta" << std::endl;
}

int updatePolicy(int pstar, int state) 
{
	std::cout << "Updating policy for state " << state << std::endl;
	return 0;
}

int main(int argc, char *argv[])
{
    int maxIt=1;
    int maxRolls=3;

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
        int numBasis = randomConfig->_numBasisX;
        float lambda = randomConfig->_lambda;
        float eta = randomConfig->_learningRate;

        for(int i=0; i<maxIt; i++) 
        {
            std::cout << "\033[1;33m\n" << "ITERATION " << i << "\033[0m" << std::endl;
            
            std::vector<SparseMatrixType> state_rolls;
            std::vector<SparseMatrixType> reward_rolls;
            std::vector<double> omega_weights;
            std::vector<double>theta(numBasis*numBasis);

            // Execute tau rollouts
            for(int tau=0; tau<maxRolls; tau++) 
            {
                std::cout << "\033[1;34m\n" << "ROLL " << tau << "\033[0m" << std::endl;
                Examples::RandomWorld world(new Examples::RandomWorldConfig(fileName), world.useOpenMPSingleNode());

                world.initialize(argc, argv);
                world.initQ();      //uncontrolled dynamics
                world.initBasis();  //center of RBF
                world.theta = theta;
                world.run();
              
                SparseMatrixType states(maxAgents,maxSteps);
                states.setFromTriplets(world._pos_spr_coeff.begin(), world._pos_spr_coeff.end());
                state_rolls.push_back(states);
                
                SparseMatrixType rewards(maxAgents,maxSteps);
                rewards.setFromTriplets(world._rwd_spr_coeff.begin(), world._rwd_spr_coeff.end());
                reward_rolls.push_back(rewards);

                //printIntMatrix(states, "State");
                printIntMatrix(rewards, "Reward");
                
                int costSum = 0;
                for (auto rewardTriplet:world._rwd_spr_coeff)
                {
                    costSum += rewardTriplet.value();
                }

                //update omega weights for current rollout
                double current_omega = world.getQoverP() * exp(-costSum/lambda); //BEA /(double)(maxSteps*maxAgents)
                std::cout << "q/p " << world.getQoverP() << "\tcostSum " << costSum << std::endl;
                std::cout << "\033[1;34m" << "omega_" << tau << ": " << "\033[0m" << current_omega << std::endl;
                omega_weights.push_back(current_omega);
                
                //update theta'
                float phi_k_x_t;
                float phi_sum = 0;
                float theta_sum = 0;
                
                //interested in one basis at a time (for each parameter of the theta vector)
                for (int k = 0; k < numBasis*numBasis; k++)
                {
                    phi_sum += world.phi_k.at(k);
                }
                for (int k = 0; k < numBasis*numBasis; k++)
                {
                    phi_k_x_t = world.phi_k.at(k);
                    phi_k_x_t /= phi_sum; //BEA normalize
                    theta.at(k) -= eta/lambda * phi_k_x_t * (current_omega - 1); //BEA
                    theta_sum += theta.at(k);
                }
                for (int k = 0; k < numBasis*numBasis; k++)
                {
                    theta.at(k) /= theta_sum; //BEA normalize
                    world.theta.at(k) = theta.at(k);
                }
                for (int k = 0; k < numBasis; k++)
                {
                    for (int k_ = 0; k_ < numBasis; k_++)
                        std::cout << (int)(world.phi_k.at(k_+numBasis*k)*100)/100.0 << "\t";
                    std::cout << std::endl;
                }
                std::cout << std::endl;
                
                for (int k = 0; k < numBasis; k++)
                {
                    for (int k_ = 0; k_ < numBasis; k_++)
                        std::cout << theta.at(k_+numBasis*k) << "\t";
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
            
            /*std::cout << "\033[1;33m" << "omega:" << "\033[0m\t";
            for (auto omega_tau:omega_weights)
                std::cout << omega_tau << "\t";
            std::cout << std::endl;*/
            
            //update Theta_{t+1} ?
        }
    }
    catch( std::exception & exceptionThrown )
    {
        std::cout << "exception thrown: " << exceptionThrown.what() << std::endl;
    }
    return 0;
}
