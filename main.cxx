
#include <RandomWorld.hxx>
#include <RandomWorldConfig.hxx>
#include <Exception.hxx>
#include <Config.hxx>
#include <eigen3/Eigen/Dense>

#include <iostream>
#include <fstream>
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
    try
    {
        if(argc>2) throw Engine::Exception("USAGE: randomWalkers [config file]");
        std::string fileName("config.xml");
        if(argc!=1) fileName = argv[1];
        
        std::ofstream outputCost;
        outputCost.open("logs/outputCost.m");

        Examples::RandomWorldConfig * randomConfig =  new Examples::RandomWorldConfig(fileName);

        randomConfig->loadFile();
        int maxIt = randomConfig->_maxIt;
        int maxRolls = randomConfig->_maxRolls;
        int maxAgents = randomConfig->_numAgents;
        int maxSteps = randomConfig->getNumSteps();
        int numBasis = randomConfig->_numBasisX;
        float lambda = randomConfig->_lambda;
        float eta = randomConfig->_learningRate;
        std::vector<double> theta(numBasis*numBasis);
        std::vector<double> average_costs;

        for(int i=0; i<maxIt; i++) 
        {
            std::cout << "\033[1;33m\n" << "ITERATION " << i << "\033[0m" << std::endl;
            
            std::vector<SparseMatrixType> state_rolls;
            std::vector<SparseMatrixType> reward_rolls;
            std::vector<double> omega_weights;
            std::vector<double> phi_k(numBasis*numBasis);
            std::vector<double> phi_k_weighted(numBasis*numBasis);
            double omega_sum = 0;
            bool goal_reached = false;
            int tau = 0;
            double last_omega = 0;
            double average_cost = 0;

            // Execute tau rollouts
            while (tau < maxRolls || !goal_reached)            
            {
                Examples::RandomWorld world(new Examples::RandomWorldConfig(fileName), world.useOpenMPSingleNode());

                world.initialize(argc, argv);
                world.initQ();      //uncontrolled dynamics
                world.initBasis();  //center of RBF
                world.theta = theta;
                world.run();
                
                SparseMatrixType rewards(maxAgents,maxSteps);               
                rewards.setFromTriplets(world._rwd_spr_coeff.begin(), world._rwd_spr_coeff.end());
                
                //total rollout cost
                float costSum = 0;
                for (auto rewardTriplet:world._rwd_spr_coeff) costSum += rewardTriplet.value();
                int final_reward = 0;
                for (int i = 0; i < maxAgents; i++) final_reward += Eigen::MatrixXf(rewards)(i, maxSteps-1);

                //update omega weights for current rollout
                double qoverp = world.getQoverP();
                double current_omega = qoverp * exp(-costSum/lambda);
                omega_weights.push_back(current_omega);                
                omega_sum += current_omega;
                
                //sum phi 
                for (int k = 0; k < numBasis*numBasis; k++)
                {
                    phi_k.at(k) += world.phi_k.at(k);
                    phi_k_weighted.at(k) += world.phi_k.at(k)*current_omega;
                }
                
                //rollout summary                
                std::cout << "Cost: " << costSum << " <- Weight: " << qoverp << " * " << exp(-costSum/lambda) << " = " << current_omega << std::endl;
                average_cost += costSum;
                
                //loop control logic
                if ((tau != 0 && last_omega != current_omega) || costSum < maxSteps*maxAgents) goal_reached = true;
                last_omega = current_omega;
                tau++;
                
                if (tau > maxRolls * 10)
                {
                    std::cout << "\033[1;31m\nMore than " << maxRolls * 10 << " rollouts were simulated in iteration " << i << " without the goal being reached.\nTry tuning lambda and the learning rate in the config.xml file.\033[0m" << std::endl;
                    exit(0);
                }
            }
            
            average_cost /= maxRolls;
            average_costs.push_back(average_cost);
            
            //update Theta_{t+1} ?
            for (int k = 0; k < numBasis*numBasis; k++)
            {
                phi_k_weighted.at(k) /= omega_sum / maxRolls;
            }
                        
            //interested in one basis at a time (for each parameter of the theta vector)
            for (int k = 0; k < numBasis*numBasis; k++)
            {
                theta.at(k) += eta/lambda/log(i+2) * (phi_k.at(k) - phi_k_weighted.at(k)); //BEA
            }

            std::cout << "\033[1;35m\n" << "Phi:" << "\033[0m" << std::endl;
            for (int k = 0; k < numBasis; k++)
            {
                for (int k_ = 0; k_ < numBasis; k_++)
                    std::cout << (int)(phi_k.at(k_+numBasis*k)*100)/100.0 << "\t";
                std::cout << std::endl;
            }
            
            std::cout << "\033[1;35m\n" << "Weighted phi:" << "\033[0m" << std::endl;
            for (int k = 0; k < numBasis; k++)
            {
                for (int k_ = 0; k_ < numBasis; k_++)
                    std::cout << (int)(phi_k_weighted.at(k_+numBasis*k)*100)/100.0 << "\t";
                std::cout << std::endl;
            }
            
            std::cout << "\033[1;35m\n" << "Current theta:" << "\033[0m" << std::endl;
            for (int k = 0; k < numBasis; k++)
            {
                for (int k_ = 0; k_ < numBasis; k_++)
                    std::cout << (int)(theta.at(k_+numBasis*k)*100)/100.0 << "\t";
                std::cout << std::endl;
            }
            
            std::cout << "\033[1;35m\n" << "Values:" << "\033[0m" << std::endl;
            double theta_phi_sum = 0;
            for (int k = 0; k < numBasis; k++)
            {
                for (int k_ = 0; k_ < numBasis; k_++)
                    if(abs(theta.at(k_+numBasis*k)*phi_k_weighted.at(k_+numBasis*k)) > theta_phi_sum) theta_phi_sum = abs(theta.at(k_+numBasis*k)*phi_k_weighted.at(k_+numBasis*k));
            }
            for (int k = 0; k < numBasis; k++)
            {
                for (int k_ = 0; k_ < numBasis; k_++)
                    std::cout << (int)(theta.at(k_+numBasis*k)*phi_k_weighted.at(k_+numBasis*k)/theta_phi_sum*1000) << "\t";
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        
        //output average costs + matlab code to plot figure
        std::cout << "AVERAGE COSTS:" << std::endl;
        outputCost << "costs = [";
        for (auto average : average_costs)
        {
            std::cout << average << "\t";
            outputCost << average << "\t";
        }
        outputCost << "];\nfig = figure;\nplot(costs, 'k');\nxlabel('Iteration');\nylabel('Average rollout cost');";
        std::cout << std::endl;
        outputCost.close();        
    }
    catch( std::exception & exceptionThrown )
    {
        std::cout << "exception thrown: " << exceptionThrown.what() << std::endl;
    }
    return 0;
}
