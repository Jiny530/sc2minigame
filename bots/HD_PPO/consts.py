

from enum import Enum
from collections import namedtuple
from sc2.ids.unit_typeid import UnitTypeId


#
#  ProxyEnv와 Actor가 주고 받는 메시지 타입
#

class CommandType(bytes, Enum):
    PING = b'\x00'
    REQ_TASK = b'\x01'
    STATE = b'\x02'
    SCORE = b'\x03'
    ERROR = b'\x04'


class ProductStrategy(Enum):
    MARINE = UnitTypeId.MARINE
    THOR = UnitTypeId.THOR
    
ProductStrategy.to_index = dict()
ProductStrategy.to_type_id = dict()

for idx, strategy in enumerate(ProductStrategy):
    ProductStrategy.to_index[strategy.value] = idx
    ProductStrategy.to_type_id[idx] = strategy.value

    
class NukeStrategy(Enum):
    OFFENSE = 0 
    DEFENSE = 1

Sample = namedtuple('Sample', 's, a, r, done, logp, value')


class MessageType(Enum):
    RESULT = 0
    EXCEPTION = 1


N_FEATURES = 6+2 #state가 6개+ProductStrategy의 2개
N_ACTIONS = len(ProductStrategy) * len(NukeStrategy)
