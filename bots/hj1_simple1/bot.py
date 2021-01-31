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
import math


class Bot(sc2.BotAI):
    """
    example v1과 유사하지만, 빌드 오더 대신, 유닛 비율을 맞추도록 유닛을 생산함
    """
    def __init__(self):
        super().__init__()
        
    def on_start(self):
        pass
        
    def patrolAttack(self, unit, actions):
        # 상대방의 좌표를 패트롤 목적지로 찍음
        # 공격상태를 한번 감지하면 바로 상대방과 나의 상대좌표를 계산 후 뒤로 물러나기
        # 일정 사거리 이상 물러나면 다시 목적지로 찍기
        # 뒤로 물러나다가 closet가 다른적으로 바뀌게 둘지? 아니면 한번 그 유닛이 죽을때까지 그 유닛만 할지?
        pass
    
    def runaway(self, u, t, dis):
        # 상대위치와 내 위치로 직선방정식 구하기
        # 상대위치에서 내 위치 방향으로 7 이상 떨어지게 하기 (상대와 내가 5거리, 그러면 내 점과 2 떨어진 점 중 직선위에 있는, 상대점과 7 떨어진 점)

        if u.x == t.x :
            if u.y < t.y:
                position = Point2((u.x,t.y-dis))
            else:
                position = Point2((u.x,t.y + dis))
        else:
            a = (t.y-u.y)/(t.x-u.x) # 기울기
            d = dis - math.sqrt((u.x-t.x)**2+(u.y-t.y)**2)
            
            p = d*d/(a**2+1)
            q = math.sqrt(p)

            if u.x < t.x : #유닛이 왼쪽이면 왼쪽으로 도망
                x = u.x - q
            else:
                x = u.x + q
            y = a*(x - u.x) + u.y
            position = Point2((x,y))

 
        return position



    

    async def on_step(self, iteration: int):
        """
        """
        actions = list() # 이번 step에 실행할 액션 목록

        cc = self.units(UnitTypeId.COMMANDCENTER).first
        cc_abilities = await self.get_available_abilities(cc)
        
        # 화염차 패트롤 무빙
        if self.can_afford(UnitTypeId.HELLION) :
            actions.append(cc.train(UnitTypeId.HELLION))

        hellions = self.units.filter(
            lambda unit: unit.type_id is UnitTypeId.HELLION
        )

        threaten = self.known_enemy_units.filter(
            lambda unit: not unit.is_flying
        )

        for unit in hellions:
            target = None
            if threaten.exists:
                target = threaten.closest_to(unit)
            
            # 상대방의 좌표를 패트롤 목적지로 찍음
            # 공격상태를 한번 감지하면 바로 상대방과 나의 상대좌표를 계산 후 뒤로 물러나기
            # 일정 사거리 이상 물러나면 다시 목적지로 찍기
            # 뒤로 물러나다가 closet가 다른적으로 바뀌게 둘지? 아니면 한번 그 유닛이 죽을때까지 그 유닛만 할지?
            if target is not None:
                
                position = None

                if unit.is_attacking or unit.distance_to(target) <= 4:
                    # 공격 했거나 가까워지면 무조건 물러나기
                    position = self.runaway(unit.position, target.position,10)
                    print(math.sqrt((unit.position.x-position.x)**2+(unit.position.y-position.y)**2))
                    actions.append(unit.move(position))

                elif unit.distance_to(target) > 9:
                    # 7 이상 벌어지면 다시 공격하러 가기
                    print("다시가자")
                    actions.append(unit.patrol(target.position))


            else:

                if unit.distance_to(self.enemy_start_locations[0]) < 10:
                    actions.append(unit.attack(self.enemy_start_locations[0]))
                else:
                    actions.append(unit.move(Point2((75,30))))

        await self.do_actions(actions)
