

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

'''
class CombatStrategy(Enum):
    OFFENSE = 0
    WAIT = 1
    DEFENSE = 2
    
CombatStrategy.to_index = dict()
CombatStrategy.to_type_id = dict()

for idx, strategy in enumerate(CombatStrategy):
    CombatStrategy.to_index[strategy.value] = idx
    CombatStrategy.to_type_id[idx] = strategy.value'''

    
class NukeStrategy(Enum):
    UPAlone = 0
    UPTogether = 1
    DownAlone = 2
    DownTogether = 3

Sample = namedtuple('Sample', 's, a, r, done, logp, value')


class MessageType(Enum):
    RESULT = 0
    EXCEPTION = 1


N_FEATURES = 5 #state가 5개
N_ACTIONS = len(NukeStrategy)
