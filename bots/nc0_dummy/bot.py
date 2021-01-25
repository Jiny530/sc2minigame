
__author__ = '박현수 (hspark8312@ncsoft.com), NCSOFT Game AI Lab'


import os

import pathlib
import pickle
import time

import nest_asyncio
import numpy as np
import sc2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython import embed
from sc2.data import Result
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.player import Bot as _Bot
from sc2.position import Point2
from termcolor import colored, cprint
from sc2.position import Point2, Point3
from enum import Enum
from random import *


nest_asyncio.apply()
class Bot(sc2.BotAI):
    """
    아무것도 하지 않는 봇 예제
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        """if self.enemy_start_locations[0].x == 95.5:
            self.patrol_pos = [point2((32.5, 35)), point2((37.5, 30)), point2((32.5, 25)), point2((27.5, 30))]
        else :
            self.patrol_pos = [point2((95.5, 35)), point2((105.5, 30)), point2((95.5, 25)), point2((90.5, 30))]
        self.a=0"""
    async def on_step(self, iteration: int):
        """
        :param int iteration: 이번이 몇 번째 스텝인지를 인자로 넘겨 줌

        매 스텝마다 호출되는 함수
        주요 AI 로직은 여기에 구현
        """
        actions = list()
        """
        if self.a ==0 and self.can_afford(next_unit) and self.time - self.evoked.get((self.bot.cc.tag, 'train'), 0) > 1.0:
                #print("00000마지막 명령을 발행한지 1초 이상 지났음")
                self.a=1
                actions.append(self.cc.train(UnitTypeId.RAVEN))

        for unit in self.units:
            actions.append(unit.patrol(patrol_pos[0],patrol_pos[1],patrol_pos[2],patrol_pos[3])
        

        # 유닛들이 수행할 액션은 리스트 형태로 만들어서,
        # do_actions 함수에 인자로 전달하면 게임에서 실행된다.
        # do_action 보다, do_actions로 여러 액션을 동시에 전달하는 
        # 것이 훨씬 빠르다."""
        
        await self.do_actions(actions)

