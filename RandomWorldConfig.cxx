
#include <RandomWorldConfig.hxx>

namespace Examples
{

RandomWorldConfig::RandomWorldConfig( const std::string & xmlFile ) : Config(xmlFile), _numAgents(0)
{
}

RandomWorldConfig::~RandomWorldConfig()
{
}

void RandomWorldConfig::loadParams()
{
	_numAgents = getParamInt( "numAgents", "value");
    _timeHorizon = getParamInt( "numSteps", "value");
	_rewardPosX = getParamInt( "reward", "xpos");
	_rewardPosY = getParamInt( "reward", "ypos");
	_rewardAreaSize = getParamInt( "reward", "size");
    _numBasis = getParamInt( "numBasis", "value");
    _basisSigma = getParamFloat( "numBasis", "sigma");
    _lambda = getParamFloat( "lambda", "value");
}
	
} // namespace Examples

