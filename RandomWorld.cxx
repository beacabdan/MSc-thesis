
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

void RandomWorld::initL(){

  for(auto pos:getBoundaries()) {
    std::vector<Engine::Point2D<int>> neihgs;
    neihgs.push_back(pos);
    if (this->getBoundaries().contains(Engine::Point2D<int> (pos._x+1, pos._y)))
      neihgs.push_back(Engine::Point2D<int> (pos._x+1, pos._y));
    if (this->getBoundaries().contains(Engine::Point2D<int> (pos._x-1, pos._y)))
      neihgs.push_back(Engine::Point2D<int> (pos._x-1, pos._y));
    if (this->getBoundaries().contains(Engine::Point2D<int> (pos._x, pos._y+1)))
      neihgs.push_back(Engine::Point2D<int> (pos._x, pos._y+1));
    if (this->getBoundaries().contains(Engine::Point2D<int> (pos._x, pos._y-1)))
      neihgs.push_back(Engine::Point2D<int> (pos._x, pos._y-1));

    for(auto p: neihgs) {
  		  this->_L_spr_coeff.push_back(Tf(
					  _ij2val(pos),           //Row is the number of the agent
					  _ij2val(p), //Col is the timestep
					  1.0/neihgs.size()
					  )
        );
    }
  }


}

int RandomWorld::_ij2val(Engine::Point2D<int> pos) {
/*
 * Convert i,j coordinates in a single scalar so we can put into a Eigen3 Sparse matrix
*/
    Engine::Size<int> s = this->getConfig().getSize();
    return pos._x*s._width + pos._y; //_height
}

int RandomWorld::_reward(Engine::Point2D<int> pos) 
{
	const RandomWorldConfig & randomConfig = (const RandomWorldConfig&)getConfig();
	if (pos.distance(Engine::Point2D<int>(randomConfig._rewardPosX, randomConfig._rewardPosY)) < randomConfig._rewardAreaSize) return 0;
    return -1;
}

void RandomWorld::step()
{
  //Step the world 
  World::step();
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
	}
}

void RandomWorld::createRasters()
{
	const RandomWorldConfig & randomConfig = (const RandomWorldConfig&)getConfig();
	
	registerDynamicRaster("resources", true);
	registerDynamicRaster("resourcesStart", true);
	getDynamicRaster("resources").setInitValues(0, 5, 0);
	getDynamicRaster("resourcesStart").setInitValues(0, 5, 0);

	for(auto index:getBoundaries())
	{
		setMaxValue("resources", index, 0);
		if (index.distance(Engine::Point2D<int>(randomConfig._rewardPosX, randomConfig._rewardPosY)) < randomConfig._rewardAreaSize)
			setMaxValue("resources", index, 1);
	}
	
	updateRasterToMaxValues("resources");
	updateRasterToMaxValues("resourcesStart");
}

float RandomWorld::L(int a, int b) {
    return 0.5;
  }

int RandomWorld::chooseRandom(std::vector<float> probs) {
 
 float dice = ( (rand() % 100) / 100.0 );
 std::cout << "\tDice is " << dice << std::endl;  

 for(int i=0; i< probs.size(); i++) {
    std::cout << "\t\tProb is " << probs[i] << std::endl;  
    dice -= probs[i];
   std::cout << "\t\t Now Dice is " << dice << std::endl;  
    if (dice < 0)
        return i;
   }

  return 0;
}

Engine::Point2D<int> RandomWorld::getAction(Engine::Agent& a)
{
  std::vector<Engine::Point2D<int>> poss;
  std::vector<float> probs;
  // Get neighs
  Engine::Point2D<int> pos = a.getPosition();
  std::cout << "I'm agent " << a.getId() << " at pos " << pos << std::endl;
  poss.push_back(pos);
  poss.push_back(Engine::Point2D<int> (pos._x+1, pos._y));
  poss.push_back(Engine::Point2D<int> (pos._x-1, pos._y)); 
  poss.push_back(Engine::Point2D<int> (pos._x, pos._y+1)); 
  poss.push_back(Engine::Point2D<int> (pos._x, pos._y-1));  

  for(auto p: poss) {
    std::cout << "\tI have a neigh at " << p << std::endl;
    if (this->getBoundaries().contains(p)) {
      float _tp = L(this->_ij2val(pos),this->_ij2val(p));
      std::cout << "\tProb is " << _tp << std::endl;
      probs.push_back(_tp);
      }
  }
  
  int index = chooseRandom(probs);
  std::cout << "\tNew pos is at " << index << " - " << poss.at(index) << std::endl;
/*  if (index > 1) {
    throw std::invalid_argument( "received negative value" );
  }
*/
  return poss.at(index);

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

