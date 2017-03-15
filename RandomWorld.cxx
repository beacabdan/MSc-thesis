
#include <RandomWorld.hxx>

#include <RandomWorldConfig.hxx>
#include <RandomAgent.hxx>
#include <DynamicRaster.hxx>
#include <Point2D.hxx>
#include <GeneralState.hxx>
#include <Logger.hxx>

namespace Examples 
{

RandomWorld::RandomWorld(Engine::Config * config, Engine::Scheduler * scheduler ) : World(config, scheduler, false)
{
}

RandomWorld::~RandomWorld()
{
}

void RandomWorld::createRasters()
{
	registerDynamicRaster("resources", true);
	registerDynamicRaster("resourcesStart", true);
	getDynamicRaster("resources").setInitValues(0, 5, 0);
	getDynamicRaster("resourcesStart").setInitValues(0, 5, 0);

	for(auto index:getBoundaries())
	{
		int value = 5; //Engine::GeneralState::statistics().getUniformDistValue(5,5);
        	setMaxValue("resources", index, value);
	}
	updateRasterToMaxValues("resources");
	updateRasterToMaxValues("resourcesStart");
}

int RandomWorld::getAction()
{
	return getId();
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

