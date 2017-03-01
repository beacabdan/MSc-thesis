
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

	Engine::Point2D<int> newPosition = agent.getPosition();

	int action = ranworld->getAction();
	
	if (action == 0)
	{
		newPosition._x += 1;
	}
	else if (action == 1)
	{
		newPosition._x -= 1;
	}
	else if (action == 2)
	{
		newPosition._y += 1;
	}
	else if (action == 3)
	{
		newPosition._y -= 1;
	}

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

