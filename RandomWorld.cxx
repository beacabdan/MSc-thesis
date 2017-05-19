
#include <RandomWorld.hxx>

#include <RandomWorldConfig.hxx>
#include <RandomAgent.hxx>
#include <DynamicRaster.hxx>
#include <GeneralState.hxx>
#include <Logger.hxx>
#include <stdexcept>

namespace Examples 
{

RandomWorld::RandomWorld(Engine::Config * config, Engine::Scheduler * scheduler ) : World(config, scheduler, true)
{
    logqpHat = 0;
}

RandomWorld::~RandomWorld()
{
}

void RandomWorld::initQ()
{
    //for all 2D pos in the world
    for(auto pos:getBoundaries())
    {
        std::vector<Engine::Point2D<int>> neighbours;
        
        neighbours.push_back(pos); //stop
        if (this->getBoundaries().contains(Engine::Point2D<int> (pos._x+1, pos._y)))
            neighbours.push_back(Engine::Point2D<int> (pos._x+1, pos._y)); //east
        if (this->getBoundaries().contains(Engine::Point2D<int> (pos._x-1, pos._y)))
            neighbours.push_back(Engine::Point2D<int> (pos._x-1, pos._y)); //west
        if (this->getBoundaries().contains(Engine::Point2D<int> (pos._x, pos._y+1)))
            neighbours.push_back(Engine::Point2D<int> (pos._x, pos._y+1)); //south
        if (this->getBoundaries().contains(Engine::Point2D<int> (pos._x, pos._y-1)))
            neighbours.push_back(Engine::Point2D<int> (pos._x, pos._y-1)); //north

        //store [pos, pos', p(pos'|pos)] for all legal pos'
        for(auto neighbour:neighbours)
            this->_q.push_back(Tf(
                _ij2val(pos),
                _ij2val(neighbour),
                1.0/neighbours.size()
                ));
    }
}

float RandomWorld::getQ(Engine::Point2D<int> source, Engine::Point2D<int> target)
{
    //find uncontrolled probability
    for (auto q:_q)
        if (_ij2val(source) == q.row() && _ij2val(target) == q.col())
            return q.value();
            
    //TODO: no need to check if neighbours, if not q(x'_i|x_i) = 0
    return 0.0;
}

void RandomWorld::initBasis()
{
    //get parameters from config.xml file
    const RandomWorldConfig & randomConfig = (const RandomWorldConfig&)getConfig();    
    int width = getBoundaries()._size._width;
    int height = getBoundaries()._size._height;
    int numBasis = randomConfig._numBasisX;
    
    //compute positions of basis
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

    //initialize theta as phi_k=0 for all k
    phi_k = std::vector<float>(numBasis*numBasis);

    //store [x][y][k][phi]
    for(auto index:getBoundaries())
    {
        _phi.push_back(getPhiOfPos(index));
    }
}

// Convert i,j coordinates in a single scalar so we can put into a Eigen3 Sparse matrix
int RandomWorld::_ij2val(Engine::Point2D<int> pos) {
    Engine::Size<int> s = this->getConfig().getSize();
    return pos._x*s._width + pos._y;
}

int RandomWorld::_ji2val(Engine::Point2D<int> pos) {
    Engine::Size<int> s = this->getConfig().getSize();
    return pos._y*s._width + pos._x;
}

Engine::Point2D<int> RandomWorld::_val2ij(int pos) {
    Engine::Size<int> s = this->getConfig().getSize();
    return Engine::Point2D<int>((pos-pos%s._width)/s._width, pos%s._width);
}

int RandomWorld::_reward(Engine::Point2D<int> pos) 
{
    float reward = 0;
    float penalty = 1;
    
	const RandomWorldConfig & randomConfig = (const RandomWorldConfig&)getConfig();
        
    for (int i = 0; i < randomConfig._numRewards; i++)
        if (pos.distance(Engine::Point2D<int>(randomConfig._rewardPositions.at(i)._x, randomConfig._rewardPositions.at(i)._y)) < randomConfig._rewardAreaSize) 
            return reward;
    
    return penalty;
}

double activation(int x, int mean_x, double sigma) {
    return 1 / std::sqrt(2 * M_PI * sigma * sigma) * std::exp(-0.5 * (x-mean_x) * (x-mean_x) / (sigma * sigma) );
}

void RandomWorld::step()
{
    //step the world 
    World::step();
    
    //get needed values from the config.xml file
    const RandomWorldConfig & randomConfig = (const RandomWorldConfig&)getConfig();
    int maxAgents = randomConfig._numAgents; 
    const int numBasis = randomConfig._numBasisX; 
    float sigma = randomConfig._basisSigma; 
    int maxTime = randomConfig._timeHorizon-1;
    
    //for each agent
	for(auto it=this->beginAgents(); it!=this->endAgents(); it++)
	{
        //get current agent
		if(!(*it)->exists()) continue;
		Engine::Agent * a = (Engine::Agent *) it->get();
        
        //store [agent, timestep, position]
		this->_pos_spr_coeff.push_back(T(
            stoi(a->getId()),
            this->getCurrentTimeStep(),
            this->_ij2val(a->getPosition())
            ));

        //store [agent, timestep, cost]
		this->_rwd_spr_coeff.push_back(T( 
            stoi(a->getId()), 
            this->getCurrentTimeStep(), 
            this->_reward(a->getPosition()) 
            ));
    }
}

//returns the ratio between q(x_{1:T}) and p_theta(x_{1:T})
double RandomWorld::getQoverP()
{
    return exp(logqpHat);
}

std::vector<float> RandomWorld::getPhiOfPos(Engine::Point2D<int> pos)
{
    //get needed values from the config.xml file
    const RandomWorldConfig & randomConfig = (const RandomWorldConfig&)getConfig();
    const int numBasis = randomConfig._numBasisX; 
    float sigma = randomConfig._basisSigma; 
    
    //vector: for each basis, activation by a single agent / pos
    std::vector<float> basisActivation(numBasis*numBasis);
    int basisCounter = 0;
    for(auto basis:basisCenters)
    {
        //sum over the agents of the activation for each basis
        basisActivation[basisCounter] = activation(basis._x, pos._x, sigma)*activation(basis._y, pos._y, sigma);
        basisCounter++;
    }
    
    return basisActivation;
}

void RandomWorld::createRasters()
{
	const RandomWorldConfig & randomConfig = (const RandomWorldConfig&)getConfig();
	
    //to represent the "goal areas"
	registerDynamicRaster("cost", true);
	getDynamicRaster("cost").setInitValues(0, 5, 0);

    //to visualize the center of the RBFs in cassandra
    registerDynamicRaster("centerRBF", true);
    getDynamicRaster("centerRBF").setInitValues(0, 5, 0);

	for(auto index:getBoundaries())
	{
        setMaxValue("cost", index, 0);
        for (int i = 0; i < randomConfig._numRewards; i++) 
            if (index.distance(Engine::Point2D<int>(randomConfig._rewardPositions.at(i)._x, randomConfig._rewardPositions.at(i)._y)) < randomConfig._rewardAreaSize) 
                setMaxValue("cost", index, 1);        
        setMaxValue("centerRBF", index, 0);
	}
	
	updateRasterToMaxValues("cost");
    updateRasterToMaxValues("centerRBF");
}

float RandomWorld::L(int a, int b) {
    return 0.3;
  }

//returns the index of the action chosen according to p_theta(x'_i|x_i)
int RandomWorld::chooseRandom(std::vector<double> transitions) {
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
    const RandomWorldConfig & randomConfig = (const RandomWorldConfig&)getConfig();
    Engine::Point2D<int> pos = a.getPosition();
    std::vector<Engine::Point2D<int>> neighbours;
    std::vector<double> p_theta;
    
    //parameters
    float lambda = randomConfig._lambda; //temperature
    int numBasis = randomConfig._numBasisX;
    numBasis *= numBasis;
    long double Z = 0; //normalization
    
    //get neighbours
    neighbours.push_back(pos);
    neighbours.push_back(Engine::Point2D<int> (pos._x+1, pos._y));
    neighbours.push_back(Engine::Point2D<int> (pos._x-1, pos._y)); 
    neighbours.push_back(Engine::Point2D<int> (pos._x, pos._y+1)); 
    neighbours.push_back(Engine::Point2D<int> (pos._x, pos._y-1));  

    //for each of the possible target cells
    for(auto targetCell : neighbours) 
    {
        //check if it's a legal action
        if (this->getBoundaries().contains(targetCell)) 
        {
            //phi(x'_i)
            std::vector<float> phi_stored = _phi.at(_ji2val(targetCell));
            
            //normalize phi(x'_i)
            float sum_phi = 0;
            for (int k = 0; k < numBasis; k++) sum_phi += phi_stored.at(k);
            //for (int k = 0; k < numBasis; k++) phi_stored.at(k) /= sum_phi; //BEA
            
            //sum phi(x_t) for all agents (for theta update)
            for (int k = 0; k < numBasis; k++) phi_k.at(k) += phi_stored.at(k);
                     
            //psi(x'_i)
            long double psi = 1;
            for (int k = 0; k < numBasis; k++) 
            {
                //WARNING: if phi * theta is too big, psi = 0
                //theta is big if sum for all t of phi is big
                //sum for all t of phi is big if T and numRolls are big
                psi *= exp( - phi_stored.at(k) * theta.at(k) / lambda );
            }
           
            //WARNING: to avoid p_theta = inf -> weight = nan
            if (isinf(psi) || psi > DBL_MAX) psi = DBL_MAX;
           
            //q(x'_i|x_i)
            float q = getQ(pos, targetCell);
            
            //p_theta(x'_i|x_i)
            double p_theta_x = psi * q;
            
            p_theta.push_back(p_theta_x);
            Z += p_theta_x;
        }
        else 
        {
            p_theta.push_back(0.0);
        }
    }
    
    //normalize p_theta probabilities 0~1
    for(std::vector<float>::size_type i = 0; i != p_theta.size(); i++)
        p_theta.at(i) /= Z;
    
    // stochastically choose an action depending on the transition probabilities
    int index = chooseRandom(p_theta);
    
    //store which \hat p_theta was used (p_hat & q_hat)
    logqpHat += log(getQ(pos, neighbours.at(index)) / p_theta.at(index));
    
    if (isinf(logqpHat) || isnan(logqpHat))
    {   
        for(std::vector<float>::size_type i = 0; i != p_theta.size(); i++) 
            std::cout << "\033[1;35m" << "p_theta(" << neighbours.at(i) << "|" << pos << "): " << "\033[0m" << p_theta.at(i) << std::endl;
        std::cout << "p at index " << index << " " << p_theta.at(index) << std::endl;
        std::cout << "log qp hat " << logqpHat << std::endl;
        std::cout << "\033[1;31m\nExiting execution because log of q over p doesn't have a value.\033[0m" << std::endl;
        exit(0);
    }
    
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
			Engine::Point2D<int> pos(3, 3); 
            //if (i % 2 == 0) pos = Engine::Point2D<int>(0, 3);
            agent->setPosition(pos);
			log_INFO(logName.str(), getWallTime() << " new agent: " << agent);
		}
	}
}

} // namespace Examples

