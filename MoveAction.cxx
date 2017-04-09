
#include <MoveAction.hxx>

#include <World.hxx>
#include <RandomAgent.hxx>
#include <GeneralState.hxx>
#include <RandomWorld.hxx>

namespace Examples
{

MoveAction::MoveAction()
{
}

MoveAction::~MoveAction()
{
} 

void MoveAction::execute( Engine::Agent & agent )
{	
	Engine::World * world = agent.getWorld();
	Examples::RandomWorld * ranworld = (Examples::RandomWorld *) agent.getWorld();

	//Engine::Point2D<int> newPosition = agent.getPosition();

    Engine::Point2D<int> newPosition = ranworld->getAction(agent);
	if(world->checkPosition(newPosition))
	{
		agent.setPosition(newPosition);
	}

	world->setValue("resourcesStart", agent.getPosition(), 0);
}

std::string MoveAction::describe() const
{
	return "MoveAction";
}

} // namespace Examples

