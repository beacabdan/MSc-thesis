
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
	return Engine::GeneralState::statistics().getUniformDistValue(0,3);
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
			oss << "RandomAgent_" << i;
			RandomAgent * agent = new RandomAgent(oss.str());
			addAgent(agent);
			agent->setRandomPosition();
	        log_INFO(logName.str(), getWallTime() << " new agent: " << agent);
		}
	}
}

} // namespace Examples

