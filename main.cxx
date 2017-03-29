
#include <RandomWorld.hxx>
#include <RandomWorldConfig.hxx>
#include <Exception.hxx>
#include <Config.hxx>
#include <eigen3/Eigen/Dense>

#include <iostream>
#include <cstdlib>

Eigen::IOFormat OctaveFmt(3, 0, ", ", ";\n", "", "", "[", "]");

int calculatePstar(std::vector<SparseMatrixType> states_rolls, std::vector<SparseMatrixType> reward_rolls, bool debug) 
{
	if (debug) 
	{
		  
		for(std::vector<SparseMatrixType>::size_type i = 0; i != states_rolls.size(); i++) 
		{
			//WARNING!!! We loose sparsity here to print!!!!
			std::cout << "**** Roll " << i << std::endl;
			std::cout << "* States: "  << std::endl; std::cout << Eigen::MatrixXd(states_rolls[i]).format(OctaveFmt) << std::endl;
			std::cout << "* Rewards: " << std::endl; std::cout << Eigen::MatrixXd(reward_rolls[i]).format(OctaveFmt) << std::endl;
			std::cout << "****" << std::endl;
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
		
		//Examples::RandomWorld world(new Examples::RandomWorldConfig(fileName), world.useSpacePartition(2));
		Examples::RandomWorldConfig * randomConfig =  new Examples::RandomWorldConfig(fileName);
		
		randomConfig->loadFile();
		int maxAgents = randomConfig->_numAgents; 
		int maxSteps = randomConfig->getNumSteps();
		std::cout << "Creating matrices of (" << maxAgents << "x" << maxSteps << ")" << std::endl; 

		for(int i=0; i<maxIt; i++) {

			std::vector<SparseMatrixType> state_rolls;
			std::vector<SparseMatrixType> reward_rolls;
			
			// Execute tau rollouts
			for(int tau=0; tau<maxRolls; tau++) 
			{

				Examples::RandomWorld world(new Examples::RandomWorldConfig(fileName), world.useOpenMPSingleNode());
				//Examples::RandomWorld world(new Examples::RandomWorldConfig(fileName), world.useSpacePartition(2));
				
				world.initialize(argc, argv);
        world.initL();

        for(auto t: world._L_spr_coeff) {
            std::cout << "\t *" << t.row() << ", " << t.col() << ", " << t.value();
        }
        SparseMatrixType L(25, 25);
				L.setFromTriplets(world._L_spr_coeff.begin(), world._L_spr_coeff.end());
        std::cout << "* L: "  << std::endl; std::cout << Eigen::MatrixXf(L).format(OctaveFmt) << std::endl;

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
