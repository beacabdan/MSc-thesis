
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
    //ask the world where to move to
	Examples::RandomWorld * ranworld = (Examples::RandomWorld *) agent.getWorld();
    Engine::Point2D<int> newPosition = ranworld->getJointAction(agent); //use getAction or getJointDistributedAction for independent behaviors
    
    //if legal position, move agent
    if(ranworld->checkPosition(newPosition)) agent.setPosition(newPosition);
}

std::string MoveAction::describe() const
{
	return "MoveAction";
}

}

