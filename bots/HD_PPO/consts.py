

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


class TacticStrategy(Enum):
    ATTACK = 0 
    RECON = 1
    NUKE = 2
    #MULE = 3


class CombatStrategy(Enum):
    OFFENSE = 0 
    DEFENSE = 1
    WAIT = 2


Sample = namedtuple('Sample', 's, a, r, done, logp, value')


class MessageType(Enum):
    RESULT = 0
    EXCEPTION = 1


N_FEATURES = 5
N_ACTIONS = len(TacticStrategy)
