
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
    _numBasisX = getParamInt( "numBasis", "xrange");
    _numBasisY = getParamInt( "numBasis", "yrange");
    _basisSigma = getParamFloat( "numBasis", "sigma");
    _lambda = getParamFloat( "lambda", "value");
    _learningRate = getParamFloat( "learningRate", "value");
}
	
} // namespace Examples

