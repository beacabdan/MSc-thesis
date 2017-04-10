
#include <RandomWorld.hxx>

#include <RandomWorldConfig.hxx>
#include <RandomAgent.hxx>
#include <DynamicRaster.hxx>
#include <GeneralState.hxx>
#include <Logger.hxx>
#include <stdexcept>

namespace Examples 
{

RandomWorld::RandomWorld(Engine::Config * config, Engine::Scheduler * scheduler ) : World(config, scheduler, false)
{
}

RandomWorld::~RandomWorld()
{
}

void RandomWorld::initL()
{
    for(auto pos:getBoundaries())
    {
        std::vector<Engine::Point2D<int>> neighbours;
        neighbours.push_back(pos);
        if (this->getBoundaries().contains(Engine::Point2D<int> (pos._x+1, pos._y)))
            neighbours.push_back(Engine::Point2D<int> (pos._x+1, pos._y));
        if (this->getBoundaries().contains(Engine::Point2D<int> (pos._x-1, pos._y)))
            neighbours.push_back(Engine::Point2D<int> (pos._x-1, pos._y));
        if (this->getBoundaries().contains(Engine::Point2D<int> (pos._x, pos._y+1)))
            neighbours.push_back(Engine::Point2D<int> (pos._x, pos._y+1));
        if (this->getBoundaries().contains(Engine::Point2D<int> (pos._x, pos._y-1)))
            neighbours.push_back(Engine::Point2D<int> (pos._x, pos._y-1));

        for(auto neighbour:neighbours) {
            this->_L_spr_coeff.push_back(Tf(
                            _ij2val(pos),       //Row is the number of the agent
                            _ij2val(neighbour), //Col is the timestep
                            1.0/neighbours.size()));
        }
    }
}

void RandomWorld::initBasis()
{
    const RandomWorldConfig & randomConfig = (const RandomWorldConfig&)getConfig();
    
    int width = getBoundaries()._size._width;
    int height = getBoundaries()._size._height;
    int numBasis = randomConfig._numBasis;
    
    int leftMargin = width % numBasis;
    int topMargin = height % numBasis;
    int xGap = (width - leftMargin) / numBasis;
    int yGap = (height - topMargin) / numBasis;
    leftMargin = (leftMargin + leftMargin % 2) / 2;
    topMargin = (topMargin + topMargin % 2) / 2;
    
    for(auto index:getBoundaries())
    {
        int x = index._x;
        int y = index._y;
        
        if ((x + 1 - leftMargin - (xGap + xGap % 2) / 2) % xGap == 0 && x + 1 - leftMargin - (xGap + xGap % 2) / 2 >= 0 && x < width - (width % numBasis) / 2 &&
            (y + 1 - topMargin - (yGap + yGap % 2) / 2) % yGap == 0 && y + 1 - topMargin - (yGap + yGap % 2) / 2 >= 0 && y < height - (height % numBasis) / 2)
        {
            setMaxValue("centerRBF", index, 5);
            setValue("centerRBF", index, 5);
            basisCenters.push_back(index);
        }
    }
}

// Convert i,j coordinates in a single scalar so we can put into a Eigen3 Sparse matrix
int RandomWorld::_ij2val(Engine::Point2D<int> pos) {
    Engine::Size<int> s = this->getConfig().getSize();
    return pos._x*s._width + pos._y; //_height
}

int RandomWorld::_reward(Engine::Point2D<int> pos) 
{
	const RandomWorldConfig & randomConfig = (const RandomWorldConfig&)getConfig();
	if (pos.distance(Engine::Point2D<int>(randomConfig._rewardPosX, randomConfig._rewardPosY)) < randomConfig._rewardAreaSize) return 0;
    return -1;
}

double activation(int x, int mean_x, double sigma) {
    return 1 / std::sqrt(2 * M_PI * sigma * sigma) * std::exp(-0.5 * (x-mean_x) * (x-mean_x) / (sigma * sigma) );
}

void RandomWorld::step()
{
    //Step the world 
    World::step();
    
    //get config
    const RandomWorldConfig & randomConfig = (const RandomWorldConfig&)getConfig();
    int maxAgents = randomConfig._numAgents; 
    int numBasis = randomConfig._numBasis; 
    
    //phi_t matrix <- how much agent m activates basis b
    std::vector<Tf> basisActivation;    
    
    //Save trajectory information
	for(auto it=this->beginAgents(); it!=this->endAgents(); it++)
	{
		if(!(*it)->exists())
		{
			continue;
		}
		Engine::Agent * a = (Engine::Agent *) it->get();
		this->_pos_spr_coeff.push_back(T(
					  stoi(a->getId()),           //Row is the number of the agent
					  this->getCurrentTimeStep(), //Col is the timestep
					  this->_ij2val(
						 a->getPosition()         //Val is the position, converted to a scalar
					  )));

		this->_rwd_spr_coeff.push_back(T(
					  stoi(a->getId()),           //Row is the number of the agent
					  this->getCurrentTimeStep(), //Col is the timestep
					  this->_reward(
						 a->getPosition()         //Val is the position, converted to a scalar
					  )));
        
        int basisCounter = 0;
        for(auto basis:basisCenters)
        {
            basisActivation.push_back(Tf(
                  stoi(a->getId()),             //Row is the number of the agent
                  basisCounter,                 //Col is the basis
                  activation(basis._x, a->getPosition()._x, 0.5)*activation(basis._y, a->getPosition()._y, 0.5)));
            basisCounter++;
        }
    }
    
    SparseMatrixType sparseM(maxAgents, numBasis*numBasis);
    sparseM.setFromTriplets(basisActivation.begin(), basisActivation.end());
    std::cout << "\033[1;35m\n" << "Phi_{t=" << getCurrentTimeStep() << "}:" << "\033[0m" << std::endl;
    Eigen::IOFormat OctaveFmt(3, 0, ", ", ";\n", "", "", "[", "]");
    std::cout << Eigen::MatrixXf(sparseM).format(OctaveFmt) << std::endl;
}

void RandomWorld::createRasters()
{
	const RandomWorldConfig & randomConfig = (const RandomWorldConfig&)getConfig();
	
	registerDynamicRaster("cost", true);
	getDynamicRaster("cost").setInitValues(0, 5, 0);
    
    registerDynamicRaster("resourcesStart", true);
	getDynamicRaster("resourcesStart").setInitValues(0, 5, 0);
    
    registerDynamicRaster("centerRBF", true);
    getDynamicRaster("centerRBF").setInitValues(0, 5, 0);

	for(auto index:getBoundaries())
	{
        setMaxValue("cost", index, 0);
		if (index.distance(Engine::Point2D<int>(randomConfig._rewardPosX, randomConfig._rewardPosY)) < randomConfig._rewardAreaSize)
			setMaxValue("cost", index, 1);
        setMaxValue("centerRBF", index, 0);
	}
	
	updateRasterToMaxValues("cost");
    updateRasterToMaxValues("resourcesStart");
    updateRasterToMaxValues("centerRBF");
}

float RandomWorld::L(int a, int b) {
    return 0.3;
  }

int RandomWorld::chooseRandom(std::vector<float> transitions) {
    float dice = ( (rand() % 100) / 100.0 );

    for(int i=0; i< transitions.size(); i++) 
    {
        dice -= transitions[i];
        if (dice < 0)
            return i;
    }

    return 0;
}

Engine::Point2D<int> RandomWorld::getAction(Engine::Agent& a)
{
    Engine::Point2D<int> pos = a.getPosition();
    std::vector<Engine::Point2D<int>> neighbours;
    std::vector<float> transitionProbabilities;
    
    // Get neighbours
    neighbours.push_back(pos);
    neighbours.push_back(Engine::Point2D<int> (pos._x+1, pos._y));
    neighbours.push_back(Engine::Point2D<int> (pos._x-1, pos._y)); 
    neighbours.push_back(Engine::Point2D<int> (pos._x, pos._y+1)); 
    neighbours.push_back(Engine::Point2D<int> (pos._x, pos._y-1));  

    // for each of the possible target cells
    for(auto targetCell : neighbours) 
    {
        // check if it's a legal action
        if (this->getBoundaries().contains(targetCell)) 
        {
            // get transition probability P(x_{t+1}|x_t)
            float _tp = L(this->_ij2val(pos), this->_ij2val(targetCell));
            transitionProbabilities.push_back(_tp);
        }
    }
    
    // stochastically choose an action depending on the transition probabilities
    int index = chooseRandom(transitionProbabilities);
    return neighbours.at(index);
}

void RandomWorld::createAgents()
{
	std::stringstream logName;
	logName << "agents_" << getId();

	const RandomWorldConfig & randomConfig = (const RandomWorldConfig&)getConfig();
	for(int i=0; i<randomConfig._numAgents; i++)
	{
		if((i%getNumTasks())==getId())
		{
			std::ostringstream oss;
			oss << i;
			RandomAgent * agent = new RandomAgent(oss.str());
			addAgent(agent);
			//If we want to apply rollout we need the agents to start always in the same positions
			//TODO: define a function to set the position on the map instead of putting all of them into the same box
			Engine::Point2D<int> pos(1,1); 
			agent->setPosition(pos);
			log_INFO(logName.str(), getWallTime() << " new agent: " << agent);
		}
	}
}

} // namespace Examples

