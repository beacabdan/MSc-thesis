
#include <RandomWorld.hxx>

#include <RandomWorldConfig.hxx>
#include <RandomAgent.hxx>
#include <DynamicRaster.hxx>
#include <GeneralState.hxx>
#include <Logger.hxx>
#include <stdexcept>
#include <string>

namespace Examples 
{

RandomWorld::RandomWorld(Engine::Config * config, Engine::Scheduler * scheduler ) : World(config, scheduler, true)
{
    logqpHat = 0;
    hareCounter = 0;
    stagCounter = 0;
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

bool isLegalAction(Engine::Point2D<int> source, Engine::Point2D<int> target)
{
    if (source._x == target._x && source._y == target._y)
        return true;
    if (source._x == target._x && abs(source._y - target._y) == 1)
        return true;
    if (source._y == target._y && abs(source._x - target._x) == 1)
        return true;
    return false;
}

void RandomWorld::initJointQ_twoAgents_old()
{
    int size = getBoundaries()._size._width * getBoundaries()._size._height;
        
    int counter = 0;
        
    //for all 2D pos in the world
    for(auto pos:getBoundaries())
    {
        for(auto pos2:getBoundaries())
        {
            std::cout << counter << std::endl;
            
            int columnCounter = 0;
            
            std::vector<int> neighboursOfCurrent;
            std::vector<float> transitionProb(size*size);
            
            //all source combinations of ag1 and ag2
            for(auto tpos:getBoundaries())
            {                
                for(auto tpos2:getBoundaries())
                {
                    //all target combinations of ag1 and ag2
                    if (isLegalAction(pos, tpos) && isLegalAction(pos2, tpos2))
                    {
                        neighboursOfCurrent.push_back(columnCounter);
                    }
                    columnCounter++;
                }
            }
            
            for (auto column:neighboursOfCurrent)
            {
                //legal actions from pos, pos2
                transitionProb[column] = 1.0/neighboursOfCurrent.size();
            }
            
            _jointQ->push_back(transitionProb);
               
            counter++; //row
        }
    }
}

void RandomWorld::initJointQ_twoAgents(std::vector<Tf> * _jointQ_sparse_common)
{
    int size = getBoundaries()._size._width * getBoundaries()._size._height;
    int counter = 0;
        
    //for all 2D pos in the world
    for(auto pos:getBoundaries())
    {
        std::vector<Engine::Point2D<int>> neighboursOfPos;            
        neighboursOfPos.push_back(pos); //stop
        if (this->getBoundaries().contains(Engine::Point2D<int> (pos._x+1, pos._y)))
            neighboursOfPos.push_back(Engine::Point2D<int> (pos._x+1, pos._y)); //east
        if (this->getBoundaries().contains(Engine::Point2D<int> (pos._x-1, pos._y)))
            neighboursOfPos.push_back(Engine::Point2D<int> (pos._x-1, pos._y)); //west
        if (this->getBoundaries().contains(Engine::Point2D<int> (pos._x, pos._y+1)))
            neighboursOfPos.push_back(Engine::Point2D<int> (pos._x, pos._y+1)); //south
        if (this->getBoundaries().contains(Engine::Point2D<int> (pos._x, pos._y-1)))
            neighboursOfPos.push_back(Engine::Point2D<int> (pos._x, pos._y-1)); //north
                
        for(auto pos2:getBoundaries())
        {            
            std::vector<int> neighboursOfCurrent;
            std::vector<float> transitionProb(size*size);
                
            std::vector<Engine::Point2D<int>> neighboursOfPos2;            
            neighboursOfPos2.push_back(pos2); //stop
            if (this->getBoundaries().contains(Engine::Point2D<int> (pos2._x+1, pos2._y)))
                neighboursOfPos2.push_back(Engine::Point2D<int> (pos2._x+1, pos2._y)); //east
            if (this->getBoundaries().contains(Engine::Point2D<int> (pos2._x-1, pos2._y)))
                neighboursOfPos2.push_back(Engine::Point2D<int> (pos2._x-1, pos2._y)); //west
            if (this->getBoundaries().contains(Engine::Point2D<int> (pos2._x, pos2._y+1)))
                neighboursOfPos2.push_back(Engine::Point2D<int> (pos2._x, pos2._y+1)); //south
            if (this->getBoundaries().contains(Engine::Point2D<int> (pos2._x, pos2._y-1)))
                neighboursOfPos2.push_back(Engine::Point2D<int> (pos2._x, pos2._y-1)); //north
            
            //all source combinations of ag1 and ag2
            for(auto tpos:neighboursOfPos)
            {                
                for(auto tpos2:neighboursOfPos2)
                {
                    neighboursOfCurrent.push_back(_ji2val(tpos)*size+_ji2val(tpos2));
                }
            }
            
            for (auto column:neighboursOfCurrent)
            {
                _jointQ_sparse_common->push_back(Tf(
                    counter,
                    column,
                    1.0/neighboursOfCurrent.size()
                    ));
            }
            counter++; //row
        }
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

float RandomWorld::getJointQ_sparse(int source, int target)
{
    //find uncontrolled probability
    for (auto q:*_jointQ_sparse)
        if (source == q.row() && target == q.col())
            return q.value();
            
    return 0.0;
}

void RandomWorld::initBasis()
{
    //get parameters from config.xml file
    const RandomWorldConfig & randomConfig = (const RandomWorldConfig&)getConfig();    
    int width = getBoundaries()._size._width;
    int height = getBoundaries()._size._height;
    int numBasis = randomConfig._numBasisX;
    int numRewards = randomConfig._numRewards;
    
    for(auto index:getBoundaries())
    {
        int x = index._x;
        int y = index._y;
        int indx = _ji2val(index);
        
        if (indx == 0+1+width || indx == width-1-1+width || indx == (height-1-1)*width+1 || indx == (height-1)*width-1-1) //4 corners
        //if (indx == 0 || indx == (width*(height/2)+width/2) || indx == width*height-1) //only 3 basis (diagonal)
        {
            setMaxValue("centerRBF", index, 5);
            setValue("centerRBF", index, 5);
            basisCenters.push_back(index);
        }
    }

    //initialize theta as phi_k=0 for all k
    phi_k = std::vector<float>(numBasis+numRewards);

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

Engine::Point2D<int> RandomWorld::_val2ji(int pos) {
    Engine::Size<int> s = this->getConfig().getSize();
    return Engine::Point2D<int>(pos%s._width, (pos-pos%s._width)/s._width);
}

float RandomWorld::_reward(Engine::Agent * ag) 
{
    const RandomWorldConfig & randomConfig = (const RandomWorldConfig&)getConfig();
    
    float reward = -1;
    float penalty = 0;
    
    float gamma1 = randomConfig._gamma;
    
    float hare = 1.25/5.0;
    float stag = 3.0/5.0;
    
    float max_val = 10.0;
    float max = ((exp(hare/gamma1) > exp(stag/gamma1)/2) ? exp(hare/gamma1) : exp(stag/gamma1)/2);
            
    for (int i = 0; i < randomConfig._numRewards; i++)
        if (ag->getPosition().distance(Engine::Point2D<int>(randomConfig._rewardPositions.at(i)._x, randomConfig._rewardPositions.at(i)._y)) < randomConfig._rewardAreaSize) 
        {
            if (i < 2)
            {
                hareCounter++;
                return max_val*(max-exp(hare/gamma1))/max; //if hare
            }
        }
        
    bool collaborate = false;
    if (ag->getPosition().distance(Engine::Point2D<int>(randomConfig._rewardPositions.at(2)._x, randomConfig._rewardPositions.at(2)._y)) < randomConfig._rewardAreaSize) 
    {
        for(auto it=this->beginAgents(); it!=this->endAgents(); it++)
        {
            Engine::Agent * a = (Engine::Agent *) it->get();
            if (a == ag) continue;
            if (a->getPosition().distance(Engine::Point2D<int>(randomConfig._rewardPositions.at(2)._x, randomConfig._rewardPositions.at(2)._y)) < randomConfig._rewardAreaSize) 
            {
                collaborate = true;
            }
        }
    }
    
    if (collaborate) 
    {
        stagCounter++;
        return max_val*(max-(exp(stag/gamma1))/2)/max; //if stag
    }
    
    return max_val*max/max;
}

double activation(int x, int mean_x, double sigma) {
    return 1 / std::sqrt(2 * M_PI * sigma * sigma) * std::exp(-0.5 * (x-mean_x) * (x-mean_x) / (sigma * sigma) );
}

void RandomWorld::step()
{
    //get needed values from the config.xml file
    const RandomWorldConfig & randomConfig = (const RandomWorldConfig&)getConfig();
    int maxAgents = randomConfig._numAgents; 
    const int numBasis = randomConfig._numBasisX; 
    float sigma = randomConfig._basisSigma; 
    int maxTime = randomConfig._timeHorizon-1;
    
    nextJointAction_distr.clear();
    
    
    for(int i = 0; i < maxAgents; i += 2)
        nextJointAction_distr.push_back(this->computeJointAction(i)); //get best next joint state
    
    World::step(); //step the world 
    
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
		this->_rwd_spr_coeff.push_back(Tf( 
            stoi(a->getId()), 
            this->getCurrentTimeStep(), 
            this->_reward(a) 
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
    int numRewards = randomConfig._numRewards;
    
    //vector: for each basis, activation by a single agent / pos
    std::vector<float> basisActivation(numBasis+numRewards);
    int basisCounter = 0;
    for(auto basis:basisCenters)
    {
        //sum over the agents of the activation for each basis
        basisActivation[basisCounter] = activation(basis._x, pos._x, sigma)*activation(basis._y, pos._y, sigma);
        basisCounter++;
    }
    
    for(int i = 0; i < numRewards; i++)
    {
        if (Engine::Point2D<int>(randomConfig._rewardPositions.at(i)) == pos)
            basisActivation[numBasis+i] = 1;
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

//returns the index of the action chosen according to p_theta(x'_i|x_i)
int RandomWorld::chooseRandom(std::vector<double> transitions)
{
    float dice = ( (rand() % 100) / 100.0 );

    for(int i=0; i< transitions.size(); i++) 
    {
        dice -= transitions[i];
        if (dice < 0)
            return i;
    }

    return 0;
}

float RandomWorld::getActivationByAllAgents(Engine::Agent& ag)
{
    const RandomWorldConfig & randomConfig = (const RandomWorldConfig&)getConfig();
    int numBasis = randomConfig._numBasisX;
    float lambda = randomConfig._lambda;
    
    float activation_ = 0;
    
    for(auto it=this->beginAgents(); it!=this->endAgents(); it++)
	{		
        Engine::Agent * a = (Engine::Agent *) it->get();
        if (a == &ag) continue;

        std::vector<float> phi_stored = _phi.at(this->_ji2val(a->getPosition()));
        for (int k = 0; k < numBasis; k++) activation_ += phi_stored.at(k) * this->theta.at(k);
    }
    
    return exp(activation_/lambda);
}

Engine::Point2D<int> RandomWorld::getJointAction(Engine::Agent& a)
{
    int size = getBoundaries()._size._width * getBoundaries()._size._height;
    
    int index = (std::stoi(a.getId())-std::stoi(a.getId())%2)/2;
    
        
    int agent1 = (nextJointAction_distr.at(index) - nextJointAction_distr.at(index) % size) / size;
    int agent2 = nextJointAction_distr.at(index) % size;
    
    if(std::stoi(a.getId())%2 == 0) return _val2ji(agent1);
    else return _val2ji(agent2);
}

 
Engine::Point2D<int> RandomWorld::getJointDistributedAction(Engine::Agent& a)
{
    const RandomWorldConfig & randomConfig = (const RandomWorldConfig&)getConfig();
    Engine::Point2D<int> pos = a.getPosition();
    std::vector<Engine::Point2D<int>> neighbours;
    std::vector<double> p_theta;
    
    //parameters
    float lambda = randomConfig._lambda; //temperature
    int numBasis = randomConfig._numBasisX;
    int numRewards = randomConfig._numRewards;
    int numAgents = randomConfig._numAgents;

    numBasis += numRewards;
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
            
            //activation of each basis by all agents (for theta update)
            for (int k = 0; k < numBasis; k++) phi_k.at(k) += phi_stored.at(k);
                                
            //get activation of each basis by all the other agents
            std::vector<float> phi_stored_ag(phi_stored.size());
            for(auto it=this->beginAgents(); it!=this->endAgents(); it++)
            {		
                Engine::Agent * ag = (Engine::Agent *) it->get();
                if (ag == &a) continue;
                Engine::Point2D<int> agpos = ag->getPosition();
                std::vector<float> phi_stored_ag_temp = _phi.at(_ji2val(agpos));
                for (int k = 0; k < numBasis; k++) phi_stored_ag.at(k) += phi_stored_ag_temp.at(k)/(float)numAgents;
            }
                                
            //psi(x'_i)
            long double psi = 1;
            for (int k = 0; k < numBasis; k++) 
            {
                psi *= exp( - (phi_stored.at(k) + phi_stored_ag.at(k)) * theta.at(k) / lambda );
            }
           
            //WARNING: to avoid p_theta = inf -> weight = nan
            if (isinf(psi) || psi > DBL_MAX) psi = DBL_MAX;
           
            //q(x'_i|x_i)
            float q = getQ(pos, targetCell);
            
            //p_theta(x'_i|x_i)
            double p_theta_x = psi * q;
            
            if(pos == Engine::Point2D<int> (0, 1) && false)
            {
                std::cout << std::endl;
                for (int k = 0; k < numBasis; k++) std::cout << ((int)(phi_stored.at(k)*10000))/10000.0 << "\t"; std::cout << std::endl;
                for (int k = 0; k < numBasis; k++) std::cout << ((int)((phi_stored.at(k)+phi_stored_ag.at(k))*10000))/10000.0 << "\t"; std::cout << std::endl;
                std::cout << p_theta_x << "(" << targetCell << "|" << pos << ") = " << psi << " * " << q << std::endl;
            }
            
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

Engine::Point2D<int> RandomWorld::getAction(Engine::Agent& a)
{
    const RandomWorldConfig & randomConfig = (const RandomWorldConfig&)getConfig();
    Engine::Point2D<int> pos = a.getPosition();
    std::vector<Engine::Point2D<int>> neighbours;
    std::vector<double> p_theta;
    
    //parameters
    float lambda = randomConfig._lambda; //temperature
    int numBasis = randomConfig._numBasisX;
    int numRewards = randomConfig._numRewards;
    
    numBasis += numRewards;
    long double Z = 0; //normalization
    
    //get neighbours
    neighbours.push_back(pos);
    neighbours.push_back(Engine::Point2D<int> (pos._x+1, pos._y));
    neighbours.push_back(Engine::Point2D<int> (pos._x-1, pos._y)); 
    neighbours.push_back(Engine::Point2D<int> (pos._x, pos._y+1)); 
    neighbours.push_back(Engine::Point2D<int> (pos._x, pos._y-1));  

    float psi_joint = this->getActivationByAllAgents(a);
    
    //for each of the possible target cells
    for(auto targetCell : neighbours) 
    {
        //check if it's a legal action
        if (this->getBoundaries().contains(targetCell)) 
        {
            //phi(x'_i)
            std::vector<float> phi_stored = _phi.at(_ji2val(targetCell));
            
            //activation of each basis by all agents (for theta update)
            for (int k = 0; k < numBasis; k++) phi_k.at(k) += phi_stored.at(k);
                                
            //psi(x'_i)
            long double psi = 1;
            for (int k = 0; k < numBasis; k++) 
            {
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

int RandomWorld::computeJointAction(int agent1id)
{    
    const RandomWorldConfig & randomConfig = (const RandomWorldConfig&)getConfig();
    int size = getBoundaries()._size._width * getBoundaries()._size._height;
    
    std::vector<double> p_theta;
    
    //parameters
    float lambda = randomConfig._lambda; //temperature
    int numBasis = randomConfig._numBasisX;
    int numRewards = randomConfig._numRewards;
    
    numBasis += numRewards;
    
    long double Z = 0; //normalization
    
    std::vector<int> neighbours;
    std::vector<Engine::Point2D<int>> agentPos;
    
    for(auto it=this->beginAgents(); it!=this->endAgents(); it++)
	{		
        Engine::Agent * a = (Engine::Agent *) it->get();
        if (std::stoi(a->getId()) == agent1id || std::stoi(a->getId()) == agent1id + 1) agentPos.push_back(a->getPosition());
    }
    
    int initialState = _ji2val(agentPos.at(0)) * size + _ji2val(agentPos.at(1));
    
    for (auto q:*_jointQ_sparse)
        if (initialState == q.row() && q.value() != 0)
            neighbours.push_back(q.col());
        
    for(auto targetCell : neighbours) 
    {
        int agent1 = (targetCell - targetCell % size) / size;
        int agent2 = targetCell % size;
        
        //phi(x'_i)
        std::vector<float> phi_stored = _phi.at(agent1);
        std::vector<float> phi_stored_2 = _phi.at(agent2);
        
        for (int i = 0; i < numBasis; i++) phi_stored.at(i) += phi_stored_2.at(i);
        if (phi_stored.at(numBasis-1) < 2) phi_stored.at(numBasis-1) = 0; //if no collaboration, activation is 0
        
        //activation of each basis by all agents (for theta update)
        for (int k = 0; k < numBasis; k++) phi_k.at(k) += phi_stored.at(k);
        
        //psi(x'_i)
        long double psi = 1;
        for (int k = 0; k < numBasis; k++) 
        {
            psi *= exp( - phi_stored.at(k) * theta.at(k) / lambda );
        }
        if (isinf(psi) || psi > DBL_MAX) psi = DBL_MAX;
        
        //q(x'_i|x_i)
        float q = this->getJointQ_sparse(initialState, targetCell);
        
        //p_theta(x'_i|x_i)
        double p_theta_x = psi * q;
        
        p_theta.push_back(p_theta_x);
        Z += p_theta_x;
    }
    
    //normalize p_theta probabilities 0~1
    for(std::vector<float>::size_type i = 0; i != p_theta.size(); i++)
        p_theta.at(i) /= Z;
    
    // stochastically choose an action depending on the transition probabilities
    int index = chooseRandom(p_theta);
    
    //store which \hat p_theta was used (p_hat & q_hat)
    logqpHat += log(this->getJointQ_sparse(initialState, neighbours.at(index)) / p_theta.at(index));
    
    if (isinf(logqpHat) || isnan(logqpHat))
    {   
        for(std::vector<float>::size_type i = 0; i != p_theta.size(); i++) 
            std::cout << "\033[1;35m" << "p_theta(" << neighbours.at(i) << "|" << initialState << "): " << "\033[0m" << p_theta.at(i) << std::endl;
        std::cout << "p at index " << index << " " << p_theta.at(index) << std::endl;
        std::cout << "log qp hat " << logqpHat << std::endl;
        std::cout << "\033[1;31m\nExiting execution because log of q over p doesn't have a value.\033[0m" << std::endl;
        exit(0);
    }
        
    nextJointAction = neighbours.at(index);
    return nextJointAction;
}

void RandomWorld::createAgents()
{
	std::stringstream logName;
	logName << "agents_" << getId();
    
    const RandomWorldConfig & randomConfig = (const RandomWorldConfig&)getConfig();
    int size = getBoundaries()._size._width;
    
	for(int i=0; i<randomConfig._numAgents; i++)
	{
		if((i%getNumTasks())==getId())
		{
			std::ostringstream oss;
			oss << i;
			RandomAgent * agent = new RandomAgent(oss.str());
			addAgent(agent);
            
            //what are the initial positions of the agents?
            Engine::Point2D<int> pos = Engine::Point2D<int>(0, (size-1)/2);
            if(agent->getId() == "1" || agent->getId() == "3" || agent->getId() == "5" || agent->getId() == "7" || agent->getId() == "9" || agent->getId() == "11" || agent->getId() == "13") pos = Engine::Point2D<int>(size-1, (size-1)/2);
            
            if(std::stoi(agent->getId())%2 == 0) agent->friend_agent = std::stoi(agent->getId()) + 1;
            else agent->friend_agent = std::stoi(agent->getId()) - 1;
            
            agent->setPosition(pos);
			log_INFO(logName.str(), getWallTime() << " new agent: " << agent);
		}
	}
}

} // namespace Examples

