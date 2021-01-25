__author__ = '이다영, 박혜진'

import os

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pathlib
import pickle
import time

from enum import Enum
import random
import math

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

from .consts import CommandType, NukeStrategy

nest_asyncio.apply()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
         #5가 state 개수, 12가 유닛종류(economy)
        self.fc1 = nn.Linear(5, 64)
        self.norm1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 64)
        self.norm2 = nn.LayerNorm(64)
        self.vf = nn.Linear(64, 1)
        #self.combat_head = nn.Linear(64, len(CombatStrategy))
        self.nuke_head = nn.Linear(64, len(NukeStrategy))
        #self.mule_head = nn.Linear(64, len(MuleStrategy))

    def forward(self, x):
        x = F.relu(self.norm1(self.fc1(x)))
        x = F.relu(self.norm2(self.fc2(x)))
        value = self.vf(x)
        #combat_logp = torch.log_softmax(self.combat_head(x), -1)
        nuke_logp = torch.log_softmax(self.nuke_head(x), -1)
        #mule_logp = torch.log_softmax(self.mule_head(x), -1)
        bz = x.shape[0]
        logp = (nuke_logp.view(bz, -1)).view(bz, -1)
        #logp = (combat_logp.view(bz, -1, 1) + nuke_logp.view(bz, 1, -1)).view(bz, -1)
        #logp = (product_logp.view(bz, 1, -1, -1) + nuke_logp.view(bz, -1, 1, -1) + mule_logp.view(bz, -1, -1, 1)).view(bz, -1)
        return value, logp

    
class NukeManager(object):
    """
    사령부 핵 담당 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
        self.pos=0
        self.dead=0
        self.ghost_pos=0
        self.enemy_pos=0
        self.ghost_tag=0
        self.nuke_time = 0
        self.stop = False

    def reset(self):
        if self.bot.enemy_start_locations[0].x == 95.5:
            self.ghost_pos = 32.5
            self.enemy_pos= 95.5
        else :
            self.ghost_pos = 95.5
            self.enemy_pos= 32.5

    # @@고스트가 우선으로 생성되게 해야할까?

    async def step(self):
        actions = list() # 이번 step에 실행할 액션 목록

        # 고스트 생산명령 내릴준비
        cc = self.bot.units(UnitTypeId.COMMANDCENTER).first
        cc_abilities = await self.bot.get_available_abilities(cc)
        ghosts = self.bot.units(UnitTypeId.GHOST)
        nuke_units = self.bot.units.tags_in(self.bot.nukeArray)
        
        # 생산
        if ghosts.amount == 0:
            actions.append(cc.train(UnitTypeId.GHOST))

        if ghosts.amount > 0:

            if self.dead == 1 and self.bot.nuke_strategy <= 1: # 위에서 죽었는데 또 위로가면
                self.nuke_reward -= 0.01 #마이너스
            elif self.dead == 2 and self.bot.nuke_strategy >= 2:
                self.nuke_reward -= 0.01 #마이너스

            self.dead = 3 

            ghost = ghosts.first #고스트는 딱 한 개체만
            threaten = self.bot.known_enemy_units.closer_than(5, ghost.position)
            nuke_units = self.bot.units.tags_in(self.bot.nukeArray)
            # 기지와 떨어졌을때 적 발견시 은폐
            if ghost.distance_to(Point2((self.ghost_pos,30))) > 10 and threaten.amount > 0:
                if not self.stop and self.bot.nuke_strategy % 2 == 0 and nuke_units.amount > 1:
                    self.bot.nuke_reward += 0.1 # reward
                    if ravens.amount > 0:
                        for unit in nuke_units:
                            actions.append(unit.attack(ravens.closest_to(unit)))
                    else :
                        for unit in nuke_units:
                            actions.append(unit.attack(threaten.closest_to(unit)))
                else : # 고스트 혼자이면 바로 은폐 쓰기
                    actions.append(ghosts.first(AbilityId.BEHAVIOR_CLOAKON_GHOST))

            

            if self.bot.nuke_strategy <= 1 and self.pos != 3: 
                for unit in nuke_units:
                # 위치 이동
                    if unit.distance_to(Point2((self.ghost_pos,55))) > 3 and self.pos == 0:
                        actions.append(unit.move(Point2((self.ghost_pos,55)))) 
                        self.pos=1 # 올라가기

                    if unit.distance_to(Point2((self.ghost_pos,55))) == 0 and self.pos == 1:
                        actions.append(unit.move(Point2((self.enemy_pos,55))))
                        self.pos=2 # 옆으로 가기

                    if unit.distance_to(Point2((self.enemy_pos,55))) < 3:
                        self.pos=3 # 대기장소 도착
            
            # 아래로 가라
            elif self.bot.nuke_strategy >= 2 and self.pos != 3: 
                # 위치 이동
                for unit in nuke_units:
                    if unit.distance_to(Point2((self.ghost_pos,10))) > 3 and self.pos == 0:
                        actions.append(unit.move(Point2((self.ghost_pos,55)))) 
                        self.pos=1 # 올라가기

                    if unit.distance_to(Point2((self.ghost_pos,10))) == 0 and self.pos == 1:
                        actions.append(unit.move(Point2((self.enemy_pos,55))))
                        self.pos=2 # 옆으로 가기

                    if unit.distance_to(Point2((self.enemy_pos,10))) < 3:
                        self.pos=3 # 대기장소 도착

            if self.pos==3 or self.stop and ghost.idle:
                self.bot.ghost_ready = True # @@고스트 준비됐음 플레그, 핵 우선으로 생산할까??
                ghost_abilities = await self.bot.get_available_abilities(ghost)
                
                if AbilityId.TACNUKESTRIKE_NUKECALLDOWN in ghost_abilities and ghost.is_idle:
                    # 전술핵 발사 가능(생산완료)하고 고스트가 idle 상태이면, 적 본진에 전술핵 발사
                    actions.append(ghost(AbilityId.BEHAVIOR_CLOAKON_GHOST))
                    actions.append(ghost(AbilityId.TACNUKESTRIKE_NUKECALLDOWN, target=self.bot.enemy_start_locations[0]))
                    self.nuke_time = self.bot.time
                    self.stop = True
                    self.bot.nuke_reward += 0.1


        elif ghosts.amount==0 and self.dead==3 : #고스트 죽음 (amount==0)
            self.pos=0
            self.bot.die_count += 1
            self.stop = False
            self.bot.nuke_reward -= 0.1

            if nuke_units.amount > 0:
                for unit in nuke_units:
                    actions.append(unit.move(self.bot.start_location))

            # @@핵 쏘는 도중 죽으면 마이너스?
            if self.bot.time - self.nuke_time < 14.0: 
                self.bot.nuke_reward -= 0.05

            if self.bot.nuke_strategy <= 1: #윗길로 갔었으면
                self.dead = 1 #위에서죽음 표시
            else:
                self.dead = 2 #아랫길이었다면 아래서 죽음 표시
        

        return actions


class ReconManager(object):
    """
    정찰부대 운용을 담당하는 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
        self.a=0
        self.patrol_pos=list()
        self.front = None
        #self.p=queue.Queue()

    def reset(self):
        if self.bot.enemy_start_locations[0].x == 95.5:
            self.patrol_pos = [Point2((25.5, 37)), Point2((39.5, 37)), Point2((39.5, 23)), Point2((25.5, 23))]
        else :
            self.patrol_pos = [Point2((102.5, 37)), Point2((88.5, 37)), Point2((88.5, 23)), Point2((102.5, 23))]
        self.front = Point2((self.bot.start_location.x + 10, 30))

    async def step(self):
        actions = list() # 이번 step에 실행할 액션 목록

        cc = self.bot.units(UnitTypeId.COMMANDCENTER).first
        cc_abilities = await self.bot.get_available_abilities(cc)
        ravens = self.bot.units(UnitTypeId.RAVEN)
        
        if ravens.amount == 0:
            if self.bot.can_afford(UnitTypeId.RAVEN):
                # 밤까마귀가 하나도 없으면 고스트 훈련
                actions.append(cc.train(UnitTypeId.RAVEN))
            if self.bot.die_alert == 2:
                self.bot.die_alert = 1 # 리콘이 죽었다 => 다른곳에서 써먹을 플래그


        elif ravens.amount > 0:
            raven = ravens.first
            
            if self.bot.die_alert == 0 or self.bot.die_alert == 1:
                self.bot.die_alert = 2 # 리콘 현재 존재함
            
            if self.bot.is_ghost == 0:
                if self.a==0 :
                    actions.append(raven.move(self.patrol_pos[0]))
                    if raven.distance_to(self.patrol_pos[0]) < 1:
                        actions.append(raven.move(self.patrol_pos[1]))
                        self.a=1
                elif self.a==1:
                    if raven.distance_to(self.patrol_pos[1]) < 1:
                        actions.append(raven.move(self.patrol_pos[2]))
                        self.a=2
                elif self.a==2:    
                    if raven.distance_to(self.patrol_pos[2]) < 1:
                        actions.append(raven.move(self.patrol_pos[3]))
                        self.a=3
                elif self.a==3:
                    if raven.distance_to(self.patrol_pos[3]) < 1:
                        actions.append(raven.move(self.patrol_pos[0]))
                        self.a=0

            threaten = self.bot.known_enemy_units.closer_than(10, raven.position) 

            #assign 관련 플래그 설정
            if raven.distance_to(self.front) > 2: #정면방향이 아님
                if threaten.exists:
                    if threaten.amount > 5:
                        self.bot.have_to_go = 2 #적이 많으면 컴뱃 전체가 움직임
                    elif threaten.amount <= 5:
                        self.bot.have_to_go = 1 #적이 조금이면 컴뱃 일부가 움직임  
                    elif threaten.amount == 1: #적이 하나면 던지고 도망가면되니까 컴뱃x
                        self.bot.have_to_go = 0
                else: self.bot.have_to_go = 0 #적이 없으면 안움직이기(돌아가게)        

            if threaten.amount > 0:
                target = threaten.closest_to(raven.position)
                self.bot.last_pos = target.position
                #print(self.bot.last_pos)
                unit = threaten(UnitTypeId.GHOST)
                
                if unit.amount > 0:
                    #print(unit.amount)
                    self.bot.is_ghost = 1 # 핵 쏘러 옴
                    target = unit.first
                    self.bot.last_pos = target.position
                    #print(self.bot.last_pos)

                if raven.distance_to(self.front) > 5 or self.bot.is_ghost: #정면방향이 아니거나, 고스트가 있을경우만 공격
                    self.bot.enemy_alert=1 # 에너미 존재
                    self.bot.last_pos = target.position
                    pos = raven.position.towards(target.position, 5)
                    pos = await self.bot.find_placement(UnitTypeId.AUTOTURRET, pos)
                    actions.append(raven(AbilityId.BUILDAUTOTURRET_AUTOTURRET, pos))

            elif threaten.amount == 0 and self.bot.enemy_alert==1: #평범하게 적들 해치운 경우
                self.bot.enemy_alert=0 # 에너미 해치움
                raven.distance_to(self.patrol_pos[self.a])
                if self.bot.is_ghost == 1:
                    # print("유령해치움")
                    self.bot.is_ghost == 0
                #else :
                    # print("에너미해치움")
                        
            elif self.bot.is_ghost: # 정면방향 고스트였을경우
                unit = threaten(UnitTypeId.GHOST)
                if unit.amount == 0:
                    self.bot.is_ghost = 0
                    self.bot.enemy_alert=0
                    raven.distance_to(self.patrol_pos[self.a])
            
        return actions


class CombatStrategy(Enum):
    OFFENSE = 0
    DEFENSE = 1


class CombatManager(object):
    """
    일반 전투 부대 컨트롤(공격+수비)
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
    
    def reset(self):
        self.evoked = dict()  
        self.move_check = 0 #이동완료=1, 아니면 0 
        self.combat_pos = 0 #combat_units의 위치
        self.target_pos = self.bot.cc #이동 위치의 기준
        #0: defense / 1: offense - 고스트 사망 횟수에 따라 달라짐
        #self.mode = 0 #시작때는 defense

    def distance(self, pos1, pos2):
        result = math.sqrt( math.pow(pos1.x - pos2.x, 2) + math.pow(pos1.y - pos2.y, 2))
        return result

    def circle(self, target_pos):
        """
        target_pos를 중점으로해서 반지름이 3인 원으로 배치
        """
        x=[]
        y=[]

        for theta in range(0, 360):
            x.append(target_pos.x+r*math.cos(math.radians(theta)))
            y.append(target_pos.y+r*math.cos(math.radians(theta)))

    async def step(self):
        actions = list()

        ##-----변수, 그룹 등 선언-----
        '''combat_tag = self.bot.units.filter(
            lambda unit: unit.tag in self.bot.combatArray
        )'''
        #combatArray에 있는 유닛들을 units로, 탱크와 마린만(레이븐은 x)->bot으로 이동
        """combat_units = self.bot.units.filter(
            lambda unit: unit.tag in self.bot.combatArray
            and unit.type_id in [UnitTypeId.MARINE, UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED]
        )"""
        #combat_units = self.bot.units.tags_in(self.bot.combatArray)
        #combat_units = self.bot.units.exclude_type([UnitTypeId.COMMANDCENTER, UnitTypeId.RAVEN, UnitTypeId.GHOST, UnitTypeId.MULE])
        
        #cc = self.bot.units(UnitTypeId.COMMANDCENTER).first->bot으로 이동
        enemy_cc = self.bot.enemy_start_locations[0]  # 적 시작 위치
        cc_abilities = await self.bot.get_available_abilities(self.bot.cc)
        mule = self.bot.units(UnitTypeId.MULE)
        mule_pos = self.bot.cc.position.towards(enemy_cc.position, -5)
        #다친 기계 유닛
        wounded_units = self.bot.units.filter(
            lambda u: u.is_mechanical and u.health_percentage < 0.5
        )  # 체력이 100% 이하인 유닛 검색

        #방어시 집결 위치
        #defense_position = self.bot.start_location + 0.25 * (enemy_cc.position - self.bot.start_location)
        def_pos1 = self.bot.start_location + 0.25 * (enemy_cc.position - self.bot.start_location)
        def_pos2 = self.bot.start_location + 0.5 * (enemy_cc.position - self.bot.start_location)
        def_pos3 = self.bot.start_location + 0.75 * (enemy_cc.position - self.bot.start_location)
        
        wait_position = self.bot.start_location + 0.5 * (enemy_cc.position - self.bot.start_location)
        
        #얼마나 많은 유닛이 공격중인가?
        attacking_units = self.bot.combat_units.filter(
            lambda u: u.is_attacking 
            and u.order_target in self.bot.known_enemy_units.tags
        )  # 유닛을 상대로 공격중인 유닛 검색

        ##-----모드 변경-----
        #if self.bot.die_count >= 3:
            #self.mode = 1 #고스트가 3번이상 죽으면 offense 모드로 변경
            #print("모드변경, 컴뱃전진")

        

        ##-----플래그 변경-----
        #3명 이상 공격중이면 fighting인걸로
        if attacking_units.amount > 5:
            self.bot.fighting = 1
            #print("공격 중인 유닛 수: ", attacking_units.amount)
        else: self.bot.fighting = 0
        #print("fighting: ", self.bot.fighting)

        ##-----Mule 기계 유닛 힐-----
        if self.bot.cc.health_percentage < 0.3: #cc피가 우선
            if mule.amount == 0:
                if AbilityId.CALLDOWNMULE_CALLDOWNMULE in cc_abilities:
                    # 지게로봇 생산가능하면 생산
                    actions.append(self.bot.cc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, mule_pos))
            elif mule.amount > 0:
                #actions.append(mule.first(AbilityId.REPAIR_MULE(cc)))
                actions.append(mule.first(AbilityId.EFFECT_REPAIR_MULE, self.bot.cc))
        elif wounded_units.exists: #아픈 토르가 있는가?
            #공격중인 토르가 있는가?
            wounded_tank = wounded_units.filter(
                lambda u: u.is_attacking 
                and u.order_target in self.bot.known_enemy_units.tags) 

            if wounded_tank.amount > 0:
                wounded_unit =wounded_tank.closest_to(self.bot.cc) #'공격중인' 탱크있으면 걔 우선
            else: wounded_unit =wounded_units.closest_to(self.bot.cc) #없으면 cc에 가까운 탱크 우선

            if mule.amount == 0:
                if AbilityId.CALLDOWNMULE_CALLDOWNMULE in cc_abilities:
                    # 지게로봇 생산가능하면 생산
                    actions.append(self.bot.cc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, wounded_unit.position))
            elif mule.amount > 0:
                #actions.append(mule.first(AbilityId.REPAIR_MULE(cc)))
                actions.append(mule.first(AbilityId.EFFECT_REPAIR_MULE, wounded_unit))
        else: #택틱이 뮬인데 치료할거리가 없을경우(치료 끝난 후)
            if mule.amount > 0:
                mule_unit=mule.random
                if self.bot.combat_units.exists:
                    actions.append(mule_unit.move(self.bot.combat_units.center))
                else: actions.append(mule_unit.move(mule_pos))

        
        #
        # -----유닛 명령 생성-----
        ## 마이크로 컨트롤
        #
        for unit in self.bot.units.not_structure:  # 건물이 아닌 유닛만 선택
            if unit in self.bot.combat_units:
                    ##-----타겟 설정-----
                enemy_unit = self.bot.enemy_start_locations[0]
                if self.bot.known_enemy_units.exists:
                    enemy_unit = self.bot.known_enemy_units.closest_to(unit)  # 가장 가까운 적 유닛

                # 적 사령부와 가장 가까운 적 유닛중 더 가까운 것을 목표로 설정
                if unit.distance_to(enemy_cc) < unit.distance_to(enemy_unit):
                    target = enemy_cc
                else:
                    target = enemy_unit

                ##-----유닛 스킬-----

                ##-----전투 유닛 전체-----
                if unit.type_id is not UnitTypeId.MEDIVAC:
                    if self.bot.combat_strategy == CombatStrategy.DEFENSE: #1=DEFENSE
                        self.target_pos = def_pos1
                        actions.append(unit.attack(self.target_pos))
                        if self.distance(self.bot.combat_units.center, def_pos1) < 3:
                            self.combat_pos = 1
                            self.move_check = 1
                    elif self.bot.combat_strategy == CombatStrategy.OFFENSE: #0=OFFENSE         
                            if self.combat_pos == 1 and self.move_check == 0:
                                #print("오펜스됨")
                                self.target_pos = def_pos2
                                actions.append(unit.attack(self.target_pos))
                                if self.distance(self.bot.combat_units.center, def_pos2) < 3:
                                    #print("pos1도착")
                                    self.combat_pos = 2
                                    self.move_check = 1
                            elif self.combat_pos == 2 and self.move_check == 0:
                                self.target_pos = def_pos3
                                actions.append(unit.attack(self.target_pos))
                                if self.distance(self.bot.combat_units.center, def_pos3) < 3:
                                    #print("pos2도착")
                                    self.combat_pos = 3
                                    self.move_check = 1
                            elif self.combat_pos == 3 and self.move_check == 0:
                                actions.append(unit.attack(self.target_pos))

                    ##-----해병과 불곰-----
                    if unit.type_id in (UnitTypeId.MARINE, UnitTypeId.MARAUDER):
                        #OFFENSE
                        if self.bot.combat_strategy == CombatStrategy.OFFENSE and unit.distance_to(target) < 15:
                            # 유닛과 목표의 거리가 15이하일 경우 스팀팩 사용
                            if not unit.has_buff(BuffId.STIMPACK) and unit.health_percentage > 0.5:
                                # 현재 스팀팩 사용중이 아니며, 체력이 50% 이상
                                if self.bot.time - self.evoked.get((unit.tag, AbilityId.EFFECT_STIM), 0) > 1.0:
                                    # 1초 이전에 스팀팩을 사용한 적이 없음
                                    actions.append(unit(AbilityId.EFFECT_STIM))
                                    self.evoked[(unit.tag, AbilityId.EFFECT_STIM)] = self.bot.time
                    """
                    ##-----토르-----
                    if unit.type_id in (UnitTypeId.THOR, UnitTypeId.THORAP):
                        if self.bot.known_enemy_units.exists:
                            #적에 전투순양함이 있으면 전투순양함에게 한방에 큰 딜 넣는게 우선
                            if self.bot.known_enemy_units.of_type(UnitTypeId.BATTLECRUISER).exists:
                                target = self.bot.known_enemy_units.of_type(UnitTypeId.BATTLECRUISER).closest_to(unit)
                                #print("토르 타겟: ", target)
                                if unit.type_id is UnitTypeId.THOR:
                                    order = unit(AbilityId.MORPH_THORHIGHIMPACTMODE)
                                    actions.append(order)
                            #전투순양함 없으면 광역딜
                            else:
                                if unit.type_id is UnitTypeId.THORAP:
                                    order = unit(AbilityId.MORPH_THOREXPLOSIVEMODE)
                                    actions.append(order)"""

                    
                    ##-----탱크-----
                    if unit.type_id in (UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED):
                        ##-----탱크-----
                        if unit.type_id in (UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED):
                            if self.bot.combat_strategy == CombatStrategy.DEFENSE:
                                #print("디펜스임-탱크")
                                if self.combat_pos == 1 and self.move_check == 1: #이동완료
                                    #print("초기조건 들어감")
                                    if unit.type_id == UnitTypeId.SIEGETANK:
                                        #print("탱크인거 알고 있음")
                                        order = unit(AbilityId.SIEGEMODE_SIEGEMODE)
                                        actions.append(order)
                            elif self.bot.combat_strategy == CombatStrategy.OFFENSE:
                                if self.move_check == 1: #이동완료
                                    if unit.type_id == UnitTypeId.SIEGETANK:
                                        order = unit(AbilityId.SIEGEMODE_SIEGEMODE)
                                        actions.append(order)
                                    elif unit.type_id == UnitTypeId.SIEGETANKSIEGED:
                                        self.move_check = 0
                                elif self.move_check == 0: #이동해야 
                                    if unit.type_id == UnitTypeId.SIEGETANKSIEGED:
                                        order = unit(AbilityId.UNSIEGE_UNSIEGE)
                                        actions.append(order)
                        
                        '''
                        #상태에 따라 분류
                        #적과의 거리 멀고 + 전차모드
                        if unit.distance_to(target) > 10 and unit.type_id == UnitTypeId.SIEGETANKSIEGED:
                            unit_state = 0
                        #적과의 거리 멀고 + 전차모드x
                        elif unit.distance_to(target) > 10 and unit.type_id == UnitTypeId.SIEGETANK:
                            unit_state = 1
                        #적과의 거리 가깝고 + 전차모드
                        elif unit.distance_to(target) <= 10 and unit.type_id == UnitTypeId.SIEGETANKSIEGED:
                            unit_state = 2
                        #적과의 거리 가깝고 + 전차모드x
                        elif unit.distance_to(target) <= 10 and unit.type_id == UnitTypeId.SIEGETANK:
                            unit_state = 3
                        else:
                            unit_state = 4

                        #DEFENSE
                        if self.bot.combat_strategy is CombatStrategy.DEFENSE:
                            if unit.distance_to(defense_position) < 5:
                                if unit.type_id == UnitTypeId.SIEGETANK:
                                    order = unit(AbilityId.SIEGEMODE_SIEGEMODE)
                                    actions.append(order) 
                                    #print("탱크됨")   
                                    
                        #OFFENSE        
                        elif self.bot.combat_strategy is CombatStrategy.OFFENSE:
                            #if unit.distance_to(target) > 10 and unit.type_id == UnitTypeId.SIEGETANKSIEGED:
                            #적이랑 멀고 전차모드임->이동->전차모드 해제
                            if unit_state == 0:
                                order = unit(AbilityId.UNSIEGE_UNSIEGE)
                                actions.append(order)
                            #적이랑 가깝고 전차아님->공격->전차모드 됨
                            elif unit_state == 3:
                                order = unit(AbilityId.SIEGEMODE_SIEGEMODE)
                                actions.append(order) 
                            #적이랑 멀고 전차아님(1)->그냥 target으로 가면 됨
                            #적이랑 가깝고 전차임(2)->그냥 target 때리면 됨
                            
                        #WAIT
                        elif self.bot.combat_strategy is CombatStrategy.WAIT:
                            #print("방어위치: ", defense_position)
                            #print("중앙집결: ", combat_units.center)
                            #적이 있으면 치는게 낫나? 아니면 그냥 빼는게 낫나? 탱크는 느려서...
                            #전차모드임->이동(후퇴)->전차모드 해제하고 이동
                            if unit_state == 0 or unit_state == 2:
                                #전차 해제
                                order = unit(AbilityId.UNSIEGE_UNSIEGE)
                                actions.append(order)

                            #대기 위치로 이동
                            actions.append(unit.move(wait_position))
                            #대기 위치 근처면 전차됨
                            if unit.distance_to(wait_position) < 3:
                                order = unit(AbilityId.SIEGEMODE_SIEGEMODE)
                                actions.append(order)   
                        
                        #타겟 우선순위 설정
                        if self.bot.known_enemy_units.exists:
                            #print("탱크 적 있음")
                            #타겟설정: 해병과 불곰 우선
                            if self.bot.known_enemy_units.of_type([UnitTypeId.MARINE, UnitTypeId.MARAUDER]).exists:
                                target = self.bot.known_enemy_units.of_type([UnitTypeId.MARINE, UnitTypeId.MARAUDER]).closest_to(unit)
                                #print("탱크 타겟설정됨")'''
                            
            
            ##-----비전투 유닛==메디박-----
            """
            if unit.type_id is UnitTypeId.MEDIVAC:
                ##힐을 가장 우선시
                if wounded_units.exists:
                    wounded_unit = wounded_units.closest_to(unit)  # 가장 가까운 체력이 100% 이하인 유닛
                    actions.append(unit(AbilityId.MEDIVACHEAL_HEAL, wounded_unit))  # 유닛 치료 명령
                    print(wounded_unit.name, "을 회복중")
                elif len(self.bot.combatArray) < 1 :
                    actions.append(unit.move(self.bot.cc - 5))
                else: 
                    actions.append(unit.move(self.bot.units.closest_to(unit)))"""

        return actions


class StepManager(object):
    """
    스텝 레이트 유지를 담당하는 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
        self.seconds_per_step = 0.35714  # on_step이 호출되는 주기
        self.reset()
    
    def reset(self):
        self.step = -1
        # 마지막으로 on_step이 호출된 게임 시간
        self.last_game_time_step_evoked = 0.0 

    def invalid_step(self):
        """
        너무 빠르게 on_step이 호출되지 않았는지 검사
        """
        elapsed_time = self.bot.time - self.last_game_time_step_evoked
        if elapsed_time < self.seconds_per_step:
            return True
        else:
            # print(C.blue(f'man_step_time: {elapsed_time}'))
            self.step += 1
            self.last_game_time_step_evoked = self.bot.time
            return False

'''
class ProductManager(object):
    """
    어떤 (부대의) 유닛을 생산할 것인지 결정하는 매니저(선택-생산-배치)
    ProductStrategy만들면서 잠깐 안씀!!!
    """
    def __init__(self, bot_ai):
        self.bot= bot_ai
    
    async def product(self, next_unit):
        # print("product 불러지면 안되는데 불러짐!!")
        """
        각 부대에서 필요한 유닛을 리스트로 bot에게 줌(비율은 각 매니저에서 계산)
        [combat, recon, nuke] 꼴, 매번 갱신
        이걸 보고 이중에서 어떤걸 생산할지 고르고 생산, 배치를 담당하는 매니저
        어떤걸 우선적으로 고를지는 택틱에 따라 결정
        """
        actions = list()
        cc_abilities = await self.bot.get_available_abilities(self.bot.cc)
        #print(self.bot.ghost_ready)
        #핵 우선생산
        if self.bot.ghost_ready:
            if AbilityId.BUILD_NUKE in cc_abilities:
                # 전술핵 생산 가능(자원이 충분)하면 전술핵 생산
                actions.append(self.bot.cc(AbilityId.BUILD_NUKE))
                self.bot.ghost_ready = False #고스트는 핵쏘는 중이라 준비x
        #핵 생산할거 없으면 나머지 생산        
        elif self.bot.can_afford(next_unit):
            #print("00000생산 가능")
            if self.bot.time - self.bot.evoked.get((self.bot.cc.tag, 'train'), 0) > 1.0:
                #print("00000마지막 명령을 발행한지 1초 이상 지났음")
                actions.append(self.bot.cc.train(next_unit))
                #print("00000생산명령-----:", next_unit)
                #self.bot.productorder = self.bot.productorder + 1
                #print("생산명령횟수: ", self.bot.productorder)
                self.bot.evoked[(self.bot.cc.tag, 'train')] = self.bot.time
        
        return actions     '''
                     
'''
class RatioManager(object):
    """
    부대에 따라 생성할 유닛 비율 결정하는 매니저
    ProductStrategy만들면서 잠깐 안씀!!!
    """
    def __init__(self, bot_ai):
        self.bot= bot_ai

    def next_unit_select(self, unit_counts, target_unit_counts):
        """
        비율에 따라 next_unit 리턴
        """
        print("ratio-next_unit_select 불러지면 안되는데 불러짐!!")
        target_units = np.array(list(target_unit_counts.values()))
        target_unit_ratio = target_units / (target_units.sum() + 1e-6)  # 목표로 하는 유닛 비율
        current_unit_counts = np.array([unit_counts.get(tid, 0) for tid in target_unit_counts.keys()])
        current_unit_ratio = current_unit_counts / (current_unit_counts.sum() + 1e-6)  # 현재 유닛 비율
        unit_ratio = (target_unit_ratio - current_unit_ratio).clip(0, 1)  # 목표 - 현재 유닛 비율
        
        next_unit = list(target_unit_counts.keys())[unit_ratio.argmax()]  # 가장 부족한 유닛을 다음에 훈련
        return next_unit

    def ratio(self):
        """
        부대별 비율에 따라 다음에 생산할 유닛 bot의 next_unit에 저장
        """
        # print("ratio 불러지면 안되는데 불러짐!!")
        #combat_unit_counts = dict()
        recon_unit_counts = dict()
        #nuke_unit_counts = dict()

        #COMBAT
        if self.bot.vespene > 200:
            combat_next_unit = UnitTypeId.SIEGETANK
            #combat_next_unit = UnitTypeId.THOR
        else:
            combat_next_unit = UnitTypeId.MARINE
        
        #RECON
        self.recon_target_unit_counts = {
                UnitTypeId.COMMANDCENTER: 0,  # 추가 사령부 생산 없음
                UnitTypeId.MARINE: 2,
                UnitTypeId.MARAUDER: 0, 
                UnitTypeId.REAPER: 0,
                UnitTypeId.GHOST: 0,
                UnitTypeId.HELLION: 0,
                UnitTypeId.SIEGETANK: 0,
                UnitTypeId.THOR: 0,
                UnitTypeId.MEDIVAC: 0,
                UnitTypeId.VIKINGFIGHTER: 0,
                UnitTypeId.BANSHEE: 0,
                UnitTypeId.RAVEN:1,
                UnitTypeId.BATTLECRUISER: 0,
            }
        for unit in self.bot.units:
            if unit.tag in self.bot.reconArray: # 유닛의 태그가 어레이에 포함되어있으면
                recon_unit_counts[unit.type_id] = recon_unit_counts.get(unit.type_id, 0) + 1
        
        recon_next_unit = self.next_unit_select(recon_unit_counts, self.recon_target_unit_counts)

        #self.bot.trainOrder = [combat_next_unit, recon_next_unit, nuke_next_unit]
        #trainOrder = [combat_next_unit, recon_next_unit]
        #유닛 결정
        #if self.bot.product_strategy == ProductStrategy.ATTACK: #combat생산
        #    self.bot.next_unit = trainOrder[0]
        #elif self.bot.product_strategy == ProductStrategy.RECON: #recon생산
        #    self.bot.next_unit = trainOrder[1]  
        #print("11111생산요청: ", self.bot.trainOrder)'''
     

class TrainManager(object):
    """
    CombatManager의 생산 유닛을 결정
    유령과 밤까마귀는 각 매니저에서 생산 결정
    """
    def __init__(self, bot_ai):
        self.bot= bot_ai
    
    def next_unit(self):
        if self.bot.vespene > 200:
            next_unit = UnitTypeId.SIEGETANK
        else:
            next_unit = UnitTypeId.MARINE

        return next_unit


class AssignManager(object): #뜯어고쳐야함
    """
    유닛을 부대에 배치하는 매니저
    *reconArray: recon에 들어가는 해병만 있음
    *nukeArray: 고스트만 있음
    *combatArray: combat의 해병, 탱크, 밤까마귀 있음

    *combat_units: combat의 해병, 탱크만 있음(combatArray보다 작은 개념)
    *nuke_units: 고스트만 있음=nukeArray
    """
    def __init__(self, bot_ai, *args, **kwargs):
        self.bot = bot_ai
        

    def reset(self):
        pass

    def assign(self):
        """
        유닛 타입을 보고 부대별 array에 유닛 배치
        """
        units_tag = self.bot.units.tags #전체유닛
        #and연산으로 살아있는 유닛으로만 구성
        self.bot.combatArray = self.bot.combatArray & units_tag 
        self.bot.reconArray = self.bot.reconArray & units_tag
        self.bot.nukeArray = self.bot.nukeArray & units_tag

        #이미 할당된 유닛의 태그 빼고
        units_tag = units_tag - self.bot.combatArray - self.bot.reconArray - self.bot.nukeArray


        #------유닛 타입에 따라 array 배정(레이븐은 패스)-----
        for tag in units_tag:
            unit = self.bot.units.find_by_tag(tag)
            if unit.type_id is UnitTypeId.RAVEN: #레이븐은 pass
                pass
            elif unit.type_id is UnitTypeId.NUKE: #핵은 pass
                pass
            elif unit.type_id is UnitTypeId.GHOST: #고스트는 nuke
                self.bot.nukeArray.add(unit.tag)
            elif unit.type_id in (UnitTypeId.SIEGETANKSIEGED, UnitTypeId.THORAP): #탱크(변신)는 컴뱃
                self.bot.combatArray.add(unit.tag)
            elif unit.type_id in (UnitTypeId.SIEGETANK, UnitTypeId.THORAP): #탱크는 컴뱃
                self.bot.combatArray.add(unit.tag)
            elif unit.type_id is UnitTypeId.MARINE: #마린도 컴뱃으로
                self.bot.combatArray.add(unit.tag)


        ##------플래그에 따라 array 배정------
        #print("fighting: ", self.bot.fighting, "/ have_togo: ", self.bot.have_to_go)
        #print("fighting이 0이고 have_togo가 1,2면 이동, fighting이 1이면 돌아옴")
        #print("nuke:", len(self.bot.nukeArray))
        #print("recon:", len(self.bot.reconArray))
        #print("combat:", len(self.bot.combatArray))
        cc = self.bot.units(UnitTypeId.COMMANDCENTER).first
        #nuke_units = self.bot.units.tags_in(self.bot.nukeArray)
        #combat_units = self.bot.units.tags_in(self.bot.combatArray)
        #if len(self.bot.combatArray) >= 10:
            #rint("에러확인1:",combat_units(UnitTypeId.MARINE).closest_to(cc).tag)
        #if len(self.bot.nukeArray) >= 1:
            #print("에러확인2:",nuke_units(UnitTypeId.MARINE).amount)

        if self.bot.fighting == 0: #컴뱃이 싸우고 있지 않음
            #------recon-----
            if self.bot.have_to_go == 1: #일부만 와라
                if len(self.bot.combatArray) >= 10: #그때 컴뱃에 10명은 넘어야(중간에 지켜야해서)
                    while len(self.bot.reconArray) == 5: #recon이 5명 될때까지
                        tag = self.bot.combat_units(UnitTypeId.MARINE).closest_to(cc).tag #해병중 cc에 가까운애 골라내기
                        self.bot.reconArray.add(tag) #recon으로 이동
                        self.bot.combatArray.remove(tag)
                        #print("이동완료1", len(self.bot.reconArray), "/", len(self.bot.combatArray))
                """
                #해병 5명 뽑기
                for tag in self.bot.combatArray:
                    unit = self.bot.units.find_by_tag(tag)
                    if unit.type_id is UnitTypeId.MARINE: #해병인가?
                        if len(self.bot.reconArray) <=5: #5명 이내인가?
                            self.bot.reconArray.add(unit.tag) #그럼 레콘으로 이동
                            break"""

            elif self.bot.have_to_go == 2: #전체가 와라
                for tag in self.bot.combatArray:
                    self.bot.reconArray.add(tag) #combat의 전부를 recon으로
                    #print("이동완료2", len(self.bot.reconArray), "/", len(self.bot.combatArray))
                self.bot.combatArray.clear() #그리고 combat 전체삭제

            #-----nuke-----
            #고스트가 새로 생성되어야하고 + 전략이 여러명이면 배정(해병)
            if  self.bot.units(UnitTypeId.GHOST).amount == 0 and self.bot.nuke_strategy%2==1:
                if self.bot.die_count <= 3: #combat이 defense인 상태
                    if len(self.bot.combatArray) >= 10: #그때 컴뱃에 10명은 넘어야(중간에 지켜야해서)
                        while self.bot.nuke_units(UnitTypeId.MARINE).amount == 5:
                            tag = self.bot.combatArray.pop() #랜덤하게 뽑음(제거)
                            self.bot.nukeArray.add(tag) #nuke로 이동
            
    
        elif self.bot.fighting == 1: #컴뱃이 싸우고 있음
            if len(self.bot.reconArray) >= 5: #몇명이 빠져있으면 돌아와라
                for tag in self.bot.reconArray:
                    self.bot.combatArray.add(tag) #recon의 전부를 combat으로
                    #print("돌아옴", len(self.bot.reconArray),"/", len(self.bot.combatArray))
                self.bot.reconArray.clear() #그리고 recon 전체삭제
                #print("완전돌아옴", len(self.bot.reconArray),"/", len(self.bot.combatArray))
            if self.bot.nuke_units(UnitTypeId.MARINE).amount >= 5: #몇명이 빠져있으면 돌아와라
                for tag in self.bot.nukeArray:
                    self.bot.combatArray.add(tag) #nuke의 전부를 combat으로
                    #print("돌아옴", len(self.bot.reconArray),"/", len(self.bot.combatArray))
                self.bot.nukeArray.clear() #그리고 recon 전체삭제
                



        """
        #유닛 타입에 따라 array 배정
        for tag in units_tag:
            unit = self.bot.units.find_by_tag(tag)
            if unit is None:
                pass
            #elif unit.type_id is UnitTypeId.RAVEN: #레이븐은 레콘으로
                #self.bot.reconArray.add(unit.tag)
            #elif unit.type_id is UnitTypeId.GHOST: #고스트는 누크로
                #self.bot.nukeArray.add(unit.tag)
            elif unit.type_id in (UnitTypeId.THOR, UnitTypeId.THORAP): #토르는 컴뱃으로
                self.bot.combatArray.add(unit.tag)
            elif unit.type_id is UnitTypeId.MARINE: #마린도 컴뱃으로
                self.bot.combatArray.add(unit.tag)
            else: #아무것도 아니여도 일단 컴뱃으로
                self.bot.combatArray.add(unit.tag)"""

        """
        if self.bot.product_strategy == ProductStrategy.ATTACK :
            self.bot.combatArray = self.bot.combatArray | units_tag
        elif self.bot.product_strategy == ProductStrategy.RECON:
            self.bot.reconArray = self.bot.reconArray | units_tag"""
        #elif self.bot.product_strategy == ProductStrategy.NUKE:
            #self.bot.nukeArray = self.bot.nukeArray | units_tag

        ## print("assign됨--------")

    def reassign(self):
        """
        이미 배정된 유닛을 다시 배정
        있을리 없는 유닛타입을 옮겨줌
        """
        for tag in self.bot.combatArray:
            unit = self.bot.units.find_by_tag(tag) #전체유닛중 컴뱃태그에 해당하는 유닛
            if unit is None:
                pass
            elif unit.type_id is UnitTypeId.RAVEN: #레이븐은 삭제
                self.bot.combatArray.remove(unit.tag)
            elif unit.type_id is UnitTypeId.GHOST: #고스트는 옮기기
                self.bot.combatArray.pop(unit.tag)
                self.bot.nukeArray.add(unit.tag)

        for tag in self.bot.nukeArray:
            unit = self.bot.units.find_by_tag(tag) #전체유닛중 누크태그에 해당하는 유닛
            if unit is None:
                pass
            elif unit.type_id is UnitTypeId.RAVEN: #레이븐은 삭제
                self.bot.nukeArray.remove(unit.tag)

        for tag in self.bot.reconArray:
            unit = self.bot.units.find_by_tag(tag) #전체유닛중 레콘태그에 해당하는 유닛
            if unit is None:
                pass
            elif unit.type_id is UnitTypeId.RAVEN: #레이븐은 삭제
                self.bot.reconArray.remove(unit.tag)
            elif unit.type_id is UnitTypeId.GHOST: #고스트는 옮기기
                self.bot.combatArray.pop(unit.tag)
                self.bot.nukeArray.add(unit.tag)

             

        ## print("reassign됨--------")


class Bot(sc2.BotAI):
    """
    빌드 오더 대신, 유닛 비율을 맞추도록 유닛을 생산함
    공격 타이밍은 강화학습을 통해 학습
    """
    def __init__(self, step_interval=5.0, host_name='', sock=None):
        super().__init__()
        self.step_interval = step_interval
        self.host_name = host_name
        self.sock = sock
        if sock is None:
            try:
                self.model = Model()
                model_path = pathlib.Path(__file__).parent / 'model.pt'
                self.model.load_state_dict(
                    torch.load(model_path, map_location='cpu')
                )
            except Exception as exc:
                import traceback; traceback.print_exc() 

        self.step_manager = StepManager(self)
        self.combat_manager = CombatManager(self)
        self.assign_manager = AssignManager(self)
        self.recon_manager = ReconManager(self)
        self.nuke_manager = NukeManager(self)
        #self.ratio_manager = RatioManager(self)
        self.train_manager = TrainManager(self)
        #self.product_manager = ProductManager(self)
        #부대별 유닛 array
        self.combatArray = set()
        self.reconArray = set()
        self.nukeArray = set()

        self.nuke_reward = 0 
        self.nuke_strategy= 0

        #self.trainOrder=list()
        #self.next_unit = UnitTypeId.MARINE
        

        #self.nukeGo = 0 #핵 쏜 횟수-bigDamage있으면 일단 필요없음 그냥 확인용으로 남겨둠
        #self.previous_Damage = 0 #이전 틱의 토탈대미지(아래를 계산하기 위한 기록용, 매 틱 갱신)
        #self.bigDamage = 0 #한번에 500이상의 큰 대미지가 들어간 횟수(핵이 500이상)

        # 정찰부대에서 사용하는 플래그
        self.threaten=list()
        self.enemy_alert=0
        self.die_alert=0 # 레이븐 죽었을때
        self.is_ghost=0 # 적이 유령인지
        self.last_pos=(0,0) # 레이븐 실시간 위치

        # 핵 부대에서 사용하는 플래그
        self.die_count=0
        self.ghost_ready = False
        
        #self.productorder = 0 #생산명령 들어간 횟수
        #self.productdone = 0 #생산명령 수행 횟수
        self.productIng = 0 #생산명령들어가면 1, 처리되면 0

        # assign에서 사용하는 플래그(갱신은 컴뱃, 레콘에서 함)
        self.fighting = 0 #현재 컴뱃에서 싸우고 있는가? 0:No, 1:Yes
        self.have_to_go = 0 #레콘으로 옮겨야하는가? 0:No, 1:일부만, 2:전체

        
    def on_start(self):
        """
        새로운 게임마다 초기화
        """
        self.step_interval = self.step_interval
        self.last_step_time = -self.step_interval

        self.combat_strategy = CombatStrategy.DEFENSE
        self.nuke_strategy = 0 #0,1,2,3
        #self.trainOrder = [UnitTypeId.MARINE, UnitTypeId.MARINE, None] #Combat,Recon,Nuke
        #self.trainOrder = [UnitTypeId.MARINE, UnitTypeId.MARINE] #Combat,Recon,Nuke
        self.evoked = dict() #생산명령 시간 체크

        self.cc = self.units(UnitTypeId.COMMANDCENTER).first  # 전체 유닛에서 사령부 검색

        self.step_manager.reset()
        self.combat_manager.reset()
        self.assign_manager.reset()
        self.nuke_manager.reset()
        self.recon_manager.reset()
        
        self.assign_manager.assign()


        # Learner에 join
        self.game_id = f"{self.host_name}_{time.time()}"
        

    async def on_step(self, iteration: int):
        """
        :param int iteration: 이번이 몇 번째 스텝인self.assign_manager = AssignManager(self)지를 인자로 넘겨 줌
        매 스텝마다 호출되는 함수
        주요 AI 로직은 여기에 구현
        """

        if self.step_manager.invalid_step():
            return list()

        actions = list() # 이번 step에 실행할 액션 목록

        #부대별 units타입
        self.nuke_units = self.units.tags_in(self.nukeArray)
        self.combat_units = self.units.filter(
            lambda unit: unit.tag in self.combatArray
            and unit.type_id in [UnitTypeId.MARINE, UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED]
        )
        self.tank_units = self.combat_units.filter(
            lambda unit:  unit.type_id in [UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED]
        )
        #self.combat_units = self.units.tags_in(self.bot.combatArray)

        if self.time - self.last_step_time >= self.step_interval:
            #택틱 변경
            before = self.nuke_strategy
            if self.combat_units.amount > 20:
                if self.tank_units.amount >= 3:
                    self.combat_strategy = CombatStrategy(0) #OFFENSE
                    self.combat_manager.move_check = 0
            self.nuke_strategy = self.set_strategy()
            if self.units(UnitTypeId.GHOST).amount > 0:
                if before != self.nuke_strategy:
                    self.nuke_reward -= 0.001
                else :
                    self.nuke_reward += 0.001
            ## print("-------생산택틱: ", self.product_strategy)
            #nuke reward 초기화
            self.nuke_reward = 0
            

            self.assign_manager.reassign() #이상하게 배치된 경우 있으면 제배치

            """
            if self.state.score.total_damage_dealt_life - self.previous_Damage > 500: #한번에 500이상의 대미지를 주었다면
                self.bigDamage += 1
                # print("한방딜 ㄱ: ", self.bigDamage, "딜량: ", self.state.score.total_damage_dealt_life - self.previous_Damage)
            self.previous_Damage = self.state.score.total_damage_dealt_life #갱신 """
            
            self.last_step_time = self.time
        

        #생산 명령이 처리되었다면
        if self.productIng == 0: 
            #self.ratio_manager.ratio() #next_unit 바꿔주고

            #생산
            #actions += await self.product_manager.product(self.next_unit) #생산
            actions += await self.train_action() #생산
            self.productIng = 1 #생산명령 들어갔다고 바꿔줌

        #생산 명령이 들어갔다면
        elif self.productIng == 1:
            #대기
            #await self.on_unit_created(self.next_unit) #명령넣은게 생산될때까지 기다림
            #self.productdone = self.productdone + 1
            ## print("생산명령 수행횟수: ", self.productdone) #엄청큼 문제있음
            ## print(self.next_unit)

            #배치
            self.assign_manager.assign()
            self.productIng = 0 #생산명령 수행했다고 바꿔줌 


        #print(self.nuke_strategy)
        #actions += await self.attack_team_manager.step()
        
        #actions += await self.product_manager.product(self.next_unit) #생산
        actions += await self.recon_manager.step() 
        actions += await self.combat_manager.step()   
        
        actions += await self.nuke_manager.step()
        
        ## -----명령 실행-----
        await self.do_actions(actions)
        ## print(self.on_unit_created)
        
        ## print("한거: ",actions)


    async def train_action(self):
        #
        # 사령부 명령 생성
        #
        actions = list()
        next_unit = self.train_manager.next_unit()

        cc_abilities = await self.get_available_abilities(self.cc)
        #핵 우선생산
        if self.ghost_ready:
            if AbilityId.BUILD_NUKE in cc_abilities:
                # 전술핵 생산 가능(자원이 충분)하면 전술핵 생산
                actions.append(self.cc(AbilityId.BUILD_NUKE))
                self.ghost_ready = False #고스트는 핵쏘는 중이라 준비x
        elif self.can_afford(next_unit):
            if self.time - self.evoked.get((self.cc.tag, 'train'), 0) > 1.0:
                # 해당 유닛 생산 가능하고, 마지막 명령을 발행한지 1초 이상 지났음
                actions.append(self.cc.train(next_unit))
                self.evoked[(self.cc.tag, 'train')] = self.time
        return actions


    def set_strategy(self):
        #
        # 특징 추출
        #
        state = np.zeros(5, dtype=np.float32)
        state[0] = self.cc.health_percentage
        state[1] = min(1.0, self.minerals / 1000)
        state[2] = min(1.0, self.vespene / 1000)
        state[3] = min(1.0, self.time / 360)
        state[4] = min(1.0, self.state.score.total_damage_dealt_life / 2500)
        #state[5] = self.nuke_reward
        '''
        state[6] = len(self.combatArray) #combat 부대의 유닛 수 - combat결정용 state
        for unit in self.units.not_structure:
            if unit in self.units.tags_in(self.combatArray):
                if unit.type_id is not UnitTypeId.MULE:
                    #토르 변신중이여도 토르로 취급
                    if unit.type_id is UnitTypeId.THORAP:
                        state[6 + ProductStrategy.to_index[UnitTypeId.THOR]] += 1
                    else:
                        state[6 + ProductStrategy.to_index[unit.type_id]] += 1'''
        #state[6] = self.trainOrder #각 부대별 생산요청 유닛 - product 결정용 - 리스트라 안됨
        state = state.reshape(1, -1)

        # NN
        data = [
            CommandType.STATE,
            pickle.dumps(self.game_id),
            pickle.dumps(state.shape),
            state,
        ]
        if self.sock is not None:
            self.sock.send_multipart(data)
            data = self.sock.recv_multipart()
            value = pickle.loads(data[0])
            action = pickle.loads(data[1])
        else:
            with torch.no_grad():
                value, logp = self.model(torch.FloatTensor(state))
                value = value.item()
                action = logp.exp().multinomial(num_samples=1).item()
        #combat_strategy = CombatStrategy.to_type_id[action // len(NukeStrategy)]
        nuke_strategy = NukeStrategy(action)
        #nuke_strategy = action % len(NukeStrategy)
        # product_strategy = ProductStrategy.to_type_id(action // (len(NukeStrategy)*len(mule_strategy)))
        # nuke_strategy = NukeStrategy(action // len(NukeStrategy))
        # mule_strategy = MuleStrategy(action % len(MuleStrategy))
        return nuke_strategy.value


    def on_end(self, game_result):
        if self.sock is not None:
            #score = 1. if game_result is Result.Victory else -1.
            
            if game_result is Result.Victory:
                score = 0.5
            else: score = -0.5
            if self.nuke_reward > 0:  #한번에 큰 대미지 넣은 횟수만큼
                score += self.nuke_reward
            # print("리워드: ", score)
            self.sock.send_multipart((
                CommandType.SCORE, 
                pickle.dumps(self.game_id),
                pickle.dumps(score),
            ))
            self.sock.recv_multipart()



    '''def on_end(self, game_result):
        for tag in self.combatArray:
            # print("111전투 : ", tag)
        # print("-----")
        for tag in self.reconArray:
            # print("2222222정찰 : ", tag)
        # print("-----")
        for tag in self.nukeArray:
            # print("핵 : ", tag)
        # print("-----")'''