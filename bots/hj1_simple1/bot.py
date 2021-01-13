# python -m bots.nc_example_v5.bot --server=172.20.41.105
# kill -9 $(ps ax | grep SC2_x64 | fgrep -v grep | awk '{ print $1 }')
# kill -9 $(ps ax | grep bots.nc_example_v5.bot | fgrep -v grep | awk '{ print $1 }')
# ps aux

import os

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
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
import queue

nest_asyncio.apply()


class Bot(sc2.BotAI):
    """
    example v1과 유사하지만, 빌드 오더 대신, 유닛 비율을 맞추도록 유닛을 생산함
    """
    def __init__(self):
        super().__init__()
        
    def on_start(self):
        self.a=0
        self.enemy_alert=0
        self.die_alert=0
        self.is_ghost=0
        self.last_pos=None
        if self.enemy_start_locations[0].x == 95.5:
            self.patrol_pos = [Point2((32.5, 35)), Point2((37.5, 30)), Point2((32.5, 25)), Point2((27.5, 30))]
        else :
            self.patrol_pos = [Point2((95.5, 35)), Point2((105.5, 30)), Point2((95.5, 25)), Point2((90.5, 30))]
        
        self.p = queue.Queue() #tlqkf 왜안돼
        self.p.put(self.patrol_pos[2])
        self.p.put(self.patrol_pos[3])
        self.p.put(self.patrol_pos[0])
        
    

    async def on_step(self, iteration: int):
        """
        """
        actions = list() # 이번 step에 실행할 액션 목록

        cc = self.units(UnitTypeId.COMMANDCENTER).first
        cc_abilities = await self.get_available_abilities(cc)
        ravens = self.units(UnitTypeId.RAVEN)
        
        if ravens.amount == 0:
            if self.can_afford(UnitTypeId.RAVEN):
                # 고스트가 하나도 없으면 고스트 훈련
                actions.append(cc.train(UnitTypeId.RAVEN))
            if self.die_alert == 2:
                self.die_alert = 1 # 리콘이 죽었다 => 다른곳에서 써먹을 플래그


        elif ravens.amount > 0:
            raven = ravens.first
            self.last_pos = raven.position

            if self.die_alert == 0 or self.die_alert == 1:
                self.die_alert = 2 # 리콘 현재 존재함
            
            if self.enemy_alert == 0:
                if self.a==0 :
                    actions.append(raven.move(self.patrol_pos[0]))
                    if raven.distance_to(self.patrol_pos[0]) < 1:
                        actions.append(raven.patrol(self.patrol_pos[1]))
                        self.a=1
                elif self.a==1:
                    if raven.distance_to(self.patrol_pos[1]) < 1:
                        actions.append(raven.patrol(self.patrol_pos[2]))
                        self.a=2
                elif self.a==2:    
                    if raven.distance_to(self.patrol_pos[2]) < 1:
                        actions.append(raven.patrol(self.patrol_pos[3]))
                        self.a=3
                elif self.a==3:
                    if raven.distance_to(self.patrol_pos[3]) < 1:
                        actions.append(raven.patrol(self.patrol_pos[0]))
                        self.a=0
            
            '''
            if raven.distance_to(self.patrol_pos[0]) == 0 :
                actions.append(raven.patrol(self.patrol_pos[1]))#,self.patrol_pos[2]))#self.p))
            '''

            threaten = self.known_enemy_units.closer_than(5, raven.position)
            if threaten.amount > 0:
                self.enemy_alert=1 # 에너미 존재
                target = threaten.closest_to(raven.position)
                for unit in threaten:
                    if unit == UnitTypeId.GHOST:
                        self.is_ghost = 1 # 핵 쏘러 옴
                        target = unit
                pos = raven.position.towards(target.position, 5)
                pos = await self.find_placement(UnitTypeId.AUTOTURRET, pos)
                actions.append(raven(AbilityId.BUILDAUTOTURRET_AUTOTURRET, pos))

            elif threaten.amount ==0 and self.enemy_alert==1:
                self.enemy_alert=0 # 에너미 해치움
                if self.is_ghost == 1:
                    print("유령해치움")
                    self.is_ghost == 0
                else :
                    print("에너미해치움")
            
            
        await self.do_actions(actions)
