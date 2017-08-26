
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
        
        Examples::RandomWorldConfig * randomConfig =  new Examples::RandomWorldConfig(fileName);
        randomConfig->loadFile();

        int maxIt = randomConfig->_maxIt;
        int maxRolls = randomConfig->_maxRolls;
        int maxAgents = randomConfig->_numAgents;
        int maxSteps = randomConfig->getNumSteps();
        int numBasis = randomConfig->_numBasisX;
        int numRewards = randomConfig->_numRewards;
        int gridSize = randomConfig->getSize()._width;
        float lambda = randomConfig->_lambda;
        float eta = randomConfig->_learningRate;
        float gamma = randomConfig->_gamma;
        float sigma = randomConfig->_basisSigma;
        std::vector<Tf> _jointQ_sparse_common;
        bool jointQstored = false;
        std::vector<double> theta(numBasis+numRewards);
        std::vector<double> average_costs;
        std::vector<double> confidence_intervals_positive;
        std::vector<double> confidence_intervals_negative;
        std::vector<double> standard_dev;
        
        std::ofstream outputCost;
        std::string path = "logs/";
        std::string sim = "pairs_" + std::to_string(maxAgents) + "agents_";
        outputCost.open(path + sim + "gamma" + std::to_string((int)(gamma*10)) + "bases" + std::to_string(numBasis) + "grid" + std::to_string(gridSize*gridSize) + ".m");
        outputCost << "% lambda = " << lambda << std::endl;
        outputCost << "% eta = " << eta << std::endl;
        outputCost << "% gamma = " << gamma << std::endl;
        outputCost << std::endl;
        outputCost << "% iterations = " << maxIt << std::endl;
        outputCost << "% rollouts = " << maxRolls << std::endl;
        outputCost << "% grid size = " << gridSize << "x" << gridSize << std::endl;
        outputCost << "% num agents = " << maxAgents << std::endl;
        outputCost << "% num bases = " << numBasis << std::endl;
        outputCost << "% sigma = " << sigma << std::endl;
        outputCost << std::endl;
        
        for(int i=0; i<maxIt; i++) 
        {
            std::cout << "\033[1;33m\n" << "ITERATION " << i << "\033[0m" << std::endl;
            
            std::vector<SparseMatrixType> state_rolls;
            std::vector<SparseMatrixType> reward_rolls;
            std::vector<double> omega_weights;
            std::vector<double> phi_k(numBasis+numRewards);
            std::vector<double> phi_k_weighted(numBasis+numRewards);
            std::vector<double> rollout_costs;
            
            double omega_sum = 0;
            bool goal_reached = false;
            int tau = 0;
            double last_omega = 0;
            double average_cost = 0;
            double average_stag = 0;
            double average_hare = 0;
            int stagCounter = 0;
            
            // Execute tau rollouts
            while (tau < maxRolls || !goal_reached)            
            {
                Examples::RandomWorld world(new Examples::RandomWorldConfig(fileName), world.useOpenMPSingleNode());
                
                world.initialize(argc, argv);
                world.initQ();      //uncontrolled dynamics
                if (!jointQstored) 
                {
                    world.initJointQ_twoAgents(&_jointQ_sparse_common);
                    jointQstored = true;
                }
                world._jointQ_sparse = &_jointQ_sparse_common;
                
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
                for (int k = 0; k < numBasis+numRewards; k++)
                {
                    phi_k.at(k) += world.phi_k.at(k);
                    phi_k_weighted.at(k) += world.phi_k.at(k)*current_omega;
                }
                
                //rollout summary
                std::cout << "Iteration: " << i << ", cost: " << (int)costSum << " <- Stags: " << world.stagCounter << ", Hares: " << world.hareCounter << " -> " << current_omega << std::endl;
                average_cost += costSum;//maxSteps*maxAgents-world.stagCounter-world.hareCounter;
                average_stag += world.stagCounter;
                average_hare += world.hareCounter;
                
                rollout_costs.push_back(costSum);
                stagCounter += world.stagCounter;
                
                //loop control logic
                if ((tau != 0 && last_omega != current_omega) || costSum < maxSteps*maxAgents || true) goal_reached = true;
                last_omega = current_omega;
                tau++;
                
                if (tau > maxRolls * 10)
                {
                    std::cout << "\033[1;31m\nMore than " << maxRolls * 10 << " rollouts were simulated in iteration " << i << " without the goal being reached.\nTry tuning lambda and the learning rate in the config.xml file.\033[0m" << std::endl;
                    exit(0);
                }                    
            }
            
            average_cost /= maxRolls;
            average_stag /= maxRolls;
            average_hare /= maxRolls;
            
            confidence_intervals_positive.push_back(average_stag);
            confidence_intervals_negative.push_back(average_hare);
            average_costs.push_back(average_cost);
                            
            //update Theta_{t+1} ?
            for (int k = 0; k < numBasis+numRewards; k++)
            {
                phi_k_weighted.at(k) /= omega_sum / maxRolls;
            }
                        
            //interested in one basis at a time (for each parameter of the theta vector)
            for (int k = 0; k < numBasis+numRewards; k++)
            {
                theta.at(k) += eta*lambda/((i+1)*log(i+2)) * (phi_k.at(k) - phi_k_weighted.at(k)); //BEA
            }

            std::cout << "\033[1;35m\n" << "Phi:" << "\033[0m" << std::endl;
            for (int k = 0; k < numBasis; k++)
            {
                std::cout << (int)(phi_k.at(k)*100)/100.0 << "\t";
            }
            
            std::cout << "\033[1;35m\n" << "Weighted phi:" << "\033[0m" << std::endl;
            for (int k = 0; k < numBasis; k++)
            {
                std::cout << (int)(phi_k_weighted.at(k)*100)/100.0 << "\t";
            }
            
            std::cout << "\033[1;35m\n" << "Current theta:" << "\033[0m" << std::endl;
            for (int k = 0; k < numBasis; k++)
            {
                std::cout << (int)(theta.at(k)*100)/100.0 << "\t";
            }
            std::cout << std::endl;
            for (int k = 0; k < numRewards; k++)
            {
                std::cout << theta.at(numBasis+k) << "\t";
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
        outputCost << ";\t";
        for (auto average : confidence_intervals_positive)
        {
            outputCost << average << "\t";
        }            
        outputCost << ";\t";
        for (auto average : confidence_intervals_negative)
        {
            outputCost << average << "\t";
        }
        outputCost << ";\t";
        outputCost << "];\n% , 'k', 'LineWidth', 2);\nfig = figure;\nhold on;\nplot(costs(1,:), 'k', 'LineWidth', 2);\nyyaxis right;\nplot(costs(2,:));\nplot(costs(3,:));\nxlabel('Iteration');\nlegend('Average rollout cost', 'Average Stags', 'Average Hares');"; //ylim([0 110]);\n
        outputCost << "\nprint('" << sim << "gamma" << std::to_string((int)(gamma*10)) << "bases" << std::to_string(numBasis) << "grid" << std::to_string(gridSize*gridSize) << "','-dpng');" << std::endl;
        std::cout << std::endl;
        
        outputCost.close();     
        
        float maxavg = 0;
        for (auto average : average_costs)
            maxavg = (average > maxavg) ? average : maxavg;
        
        std::cout << std::endl;
        for (auto average : average_costs)
        {
            for (int i = 0; i < average*30.0/maxavg; i++)
                std::cout << "+";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        
        float gamma1 = randomConfig->_gamma;
        float hare = 1.25/5.0;
        float stag = 3.0/5.0;
        float max_val = 10.0;
        float max = ((exp(hare/gamma1) > exp(stag/gamma1)/2) ? exp(hare/gamma1) : exp(stag/gamma1)/2);            
        std::cout << "Hare: " << max_val*(max-exp(hare/gamma1))/max << "\nStag: " << max_val*(max-(exp(stag/gamma1))/2)/max << std::endl;
    }
    catch( std::exception & exceptionThrown )
    {
        std::cout << "exception thrown: " << exceptionThrown.what() << std::endl;
    }
    return 0;
}
